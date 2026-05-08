#!/usr/bin/env python3
"""
Shannon-Prime / freethedsp — Phase D.2 offset-finder.

Locates `is_test_enabled` in a captured fastrpc_shell_N dump and emits
the PATCH_ADDR / PATCH_OLD / PATCH_NEW values to bake into
freethedsp_s22u.c.

Usage:
    python find_is_test_enabled.py <fastrpc_shell.bin>
    python find_is_test_enabled.py --hexagon-objdump=<path> <fastrpc_shell.bin>

Algorithm:
    1. SHA-256 the dump (so we can detect firmware drift later).
    2. Verify ELF magic; bail loudly if not an ELF (means the dump
       captured the wrong page).
    3. Run hexagon-llvm-objdump --syms on the dump. Look for a symbol
       named `is_test_enabled`. If found, that's our address.
    4. If not in symtab, run --disassemble and grep for the canonical
       2-instruction body of is_test_enabled: `r0 = #0; jumpr r31`
       (Hexagon `lr` is r31). The function is short and has a distinct
       signature.
    5. Emit:
         PATCH_ADDR  = 0xXXXX (file offset of the `r0 = #0` insn)
         PATCH_OLD   = "\\xAA\\xBB\\xCC\\xDD" (the 4-byte encoding)
         PATCH_NEW   = "\\xEE\\xFF\\xGG\\xHH" (4-byte encoding of `r0 = #-1`)
         SHELL_SHA256 = "..."

Hexagon instruction encoding cheat-sheet (V66+):
    `r0 = #0`           one of:
        0x78 00 00 c0   movi r0, #0       (0x7800c000 LE)
        0x70 00 00 40   r0 = #0           (different parse class)
    `r0 = #-1`          0x78 1f ff c0   movi r0, #-1   (0x781fffc0 LE)
    `jumpr r31`         0x52 1f c0 00   jumpr r31      (0x521fc000 LE)
    Often the function is a packet-pair:
        { r0 = #0   ;   jumpr r31 }
    encoded in a single 32-bit word with the parallel-execution bit set.
    That's the form geohot's reference patch uses.
"""

from __future__ import annotations
import argparse
import hashlib
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_elf(p: Path) -> bool:
    with open(p, "rb") as f:
        magic = f.read(4)
    return magic == b"\x7fELF"


def find_objdump(override: str | None) -> str:
    if override:
        return override
    sdk = os.environ.get("HEXAGON_SDK_ROOT", r"C:\Qualcomm\Hexagon_SDK\5.5.6.0")
    ver = os.environ.get("HEXAGON_TOOLS_VER", "8.7.06")
    candidates = [
        Path(sdk) / "tools" / "HEXAGON_Tools" / ver / "Tools" / "bin" / "hexagon-llvm-objdump.exe",
        Path(sdk) / "tools" / "HEXAGON_Tools" / ver / "Tools" / "bin" / "hexagon-llvm-objdump",
        Path(sdk) / "tools" / "HEXAGON_Tools" / ver / "Tools" / "bin" / "hexagon-objdump.exe",
        Path(sdk) / "tools" / "HEXAGON_Tools" / ver / "Tools" / "bin" / "hexagon-objdump",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(
        f"Could not find hexagon-llvm-objdump under {sdk}/tools/HEXAGON_Tools/{ver}/Tools/bin/.\n"
        f"Pass --hexagon-objdump=<path> to override."
    )


def run(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, errors="replace")
    except subprocess.CalledProcessError as e:
        out = e.output or ""
    return out


def find_via_symbols(objdump: str, shell: Path) -> tuple[int, bytes] | None:
    """Look up is_test_enabled in the symbol table. Return (file_offset, 4_bytes_at_that_offset) or None."""
    out = run([objdump, "--syms", str(shell)])
    # Symbol lines look like:  00000000abcdef00 g    F .text 00000010 is_test_enabled
    for line in out.splitlines():
        if "is_test_enabled" not in line:
            continue
        parts = line.split()
        try:
            addr = int(parts[0], 16)
        except (ValueError, IndexError):
            continue
        # The address from --syms is a virtual address. We need a file
        # offset, which for a static-loaded ELF segment usually equals
        # vaddr - segment_base. Read program headers to translate.
        return _vaddr_to_file_offset(shell, addr)
    return None


def _vaddr_to_file_offset(shell: Path, vaddr: int) -> tuple[int, bytes] | None:
    """Translate ELF virtual address to file offset using program headers."""
    with open(shell, "rb") as f:
        e_ident = f.read(16)
        if e_ident[:4] != b"\x7fELF":
            return None
        ei_class = e_ident[4]  # 1=32-bit, 2=64-bit
        ei_data = e_ident[5]   # 1=little-endian
        endian = "<" if ei_data == 1 else ">"

        if ei_class == 1:  # ELF32
            f.read(2 + 2 + 4)  # e_type, e_machine, e_version
            f.read(4)          # e_entry
            e_phoff = struct.unpack(endian + "I", f.read(4))[0]
            f.read(4 + 4 + 2)  # e_shoff, e_flags, e_ehsize
            e_phentsize = struct.unpack(endian + "H", f.read(2))[0]
            e_phnum = struct.unpack(endian + "H", f.read(2))[0]
            f.seek(e_phoff)
            for _ in range(e_phnum):
                p_type   = struct.unpack(endian + "I", f.read(4))[0]
                p_offset = struct.unpack(endian + "I", f.read(4))[0]
                p_vaddr  = struct.unpack(endian + "I", f.read(4))[0]
                f.read(4)                                            # p_paddr
                p_filesz = struct.unpack(endian + "I", f.read(4))[0]
                f.read(4 + 4 + 4)                                    # p_memsz, p_flags, p_align
                if p_type == 1 and p_vaddr <= vaddr < p_vaddr + p_filesz:  # PT_LOAD
                    file_off = p_offset + (vaddr - p_vaddr)
                    f.seek(file_off)
                    bytes4 = f.read(4)
                    return file_off, bytes4
        # ELF64 path omitted for brevity — fastrpc shells have historically
        # been 32-bit Hexagon; extend here if that changes.
    return None


def find_via_disassembly(objdump: str, shell: Path) -> tuple[int, bytes] | None:
    """Fallback: scan disassembly for the `r0=#0; jumpr r31` packet pattern."""
    out = run([objdump, "-d", str(shell)])
    # Look for the 2-instruction body. Format depends on objdump's syntax;
    # we accept several common spellings.
    pat = re.compile(
        r"^\s*([0-9a-f]+):\s+([0-9a-f ]+)\s+\{ r0\.?\s*=\s*#0 ?; jumpr (?:r31|lr) \}",
        re.IGNORECASE | re.MULTILINE,
    )
    matches = pat.findall(out)
    if not matches:
        # Try the looser two-line packet form
        pat2 = re.compile(
            r"^\s*([0-9a-f]+):\s+([0-9a-f ]+)\s+r0\.?\s*=\s*#0\b.*?\n"
            r"^\s*[0-9a-f]+:\s+[0-9a-f ]+\s+jumpr\s+(?:r31|lr)\b",
            re.IGNORECASE | re.MULTILINE,
        )
        matches = pat2.findall(out)
    if not matches:
        return None
    # If multiple candidates, the user must inspect — emit all.
    print(f"[find] {len(matches)} candidate `{{r0=#0; jumpr r31}}` packets found.")
    for vaddr_hex, bytes_hex in matches:
        print(f"[find]   vaddr=0x{vaddr_hex}  bytes={bytes_hex.strip()}")
    if len(matches) > 1:
        print("[find] Multiple candidates — disambiguate with a symbol-based "
              "tool (nm, llvm-readobj --section-headers) and pick the one "
              "in .text whose surroundings match is_test_enabled's CFG.")
    vaddr_hex, bytes_hex = matches[0]
    vaddr = int(vaddr_hex, 16)
    file_off_pair = _vaddr_to_file_offset(shell, vaddr)
    if not file_off_pair:
        return None
    return file_off_pair


def emit_constants(file_off: int, old_bytes: bytes, sha: str) -> None:
    # PATCH_NEW: r0 = #-1 instead of r0 = #0. Encoding-wise on Hexagon, the
    # immediate field of the `movi` instruction is a 16-bit signed
    # placement; -1 is 0xFFFF in twos-complement, masked into the
    # encoding-specific bits. Easiest: emit a recommendation and let the
    # operator hand-encode after disassembling the chosen candidate.
    print()
    print("=" * 70)
    print("Bake the following into backends/freethedsp/freethedsp_s22u.c:")
    print("=" * 70)
    print(f"#define PATCH_ADDR    0x{file_off:08x}")
    print(f"#define PATCH_OLD     " +
          '"' + ''.join(f"\\x{b:02x}" for b in old_bytes) + '"')
    print("#define PATCH_NEW     /* TODO_HAND_ENCODE — change `r0=#0` to `r0=#-1` */")
    print(f"#define SHELL_SHA256  \"{sha}\"")
    print("=" * 70)
    print()
    print("Hand-encode PATCH_NEW: take the disassembly of the matched packet,")
    print("substitute `r0=#-1` for `r0=#0`, re-assemble with hexagon-llvm-mc, and")
    print("paste the resulting 4 bytes into PATCH_NEW.")
    print()
    print("Or, for the canonical reference encoding from geohot:")
    print('  PATCH_OLD ~ "\\x40\\x3f\\x20\\x50"  ({r0=#0; jumpr r31})')
    print('  PATCH_NEW ~ "\\x40\\x3f\\x00\\x5a"  ({r0=#-1; jumpr r31})')
    print("These specific bytes are device-firmware-specific — confirm against")
    print("our captured PATCH_OLD before pasting PATCH_NEW.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shell", type=Path, help="Path to captured fastrpc_shell_N dump")
    ap.add_argument("--hexagon-objdump", default=None, help="Path to hexagon-llvm-objdump (auto-detected by default)")
    args = ap.parse_args()

    shell: Path = args.shell.resolve()
    if not shell.exists():
        print(f"[error] shell file not found: {shell}", file=sys.stderr)
        return 1
    if not is_elf(shell):
        print(f"[error] {shell} doesn't start with ELF magic — wrong dump?", file=sys.stderr)
        return 1
    sha = sha256_of(shell)
    print(f"[info] shell sha256: {sha}")
    print(f"[info] shell size:   {shell.stat().st_size} bytes")

    objdump = find_objdump(args.hexagon_objdump)
    print(f"[info] using {objdump}")

    res = find_via_symbols(objdump, shell)
    if res:
        file_off, old_bytes = res
        print(f"[find] is_test_enabled via symbol table at file offset 0x{file_off:x}")
    else:
        print("[find] is_test_enabled not in symbol table — falling back to disassembly scan")
        res = find_via_disassembly(objdump, shell)
        if not res:
            print("[error] could not locate is_test_enabled. Manual inspection required:")
            print(f"        {objdump} -d {shell} | less")
            return 2
        file_off, old_bytes = res
        print(f"[find] candidate is_test_enabled at file offset 0x{file_off:x}")

    emit_constants(file_off, old_bytes, sha)
    return 0


if __name__ == "__main__":
    sys.exit(main())
