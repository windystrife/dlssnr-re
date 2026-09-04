<#
.SYNOPSIS
  Disassemble an AMDGPU code object using AMD's own disassembler.

.DESCRIPTION
  Drives amd_comgr_*.dll - the LLVM-based code-object manager that ships with
  every Adrenalin driver in C:\Windows\System32 - through P/Invoke. Nothing to
  install: no ROCm, no HIP SDK, no LLVM. This matters because the LLVM builds
  bundled with Visual Studio and the Android NDK are compiled WITHOUT the
  AMDGPU target, so llvm-objdump on a typical Windows box cannot decode gfx11
  or gfx12 at all.

  Feed it the raw .text bytes and a symbol table, both produced by:
      python dlssnr_bundle.py text bundle-gfx1201.bin -o text.bin --syms syms.txt

.PARAMETER Isa
  Target triple, e.g. amdgcn-amd-amdhsa--gfx1201 (RDNA4) or
  amdgcn-amd-amdhsa--gfx1100 (RDNA3).

.EXAMPLE
  .\disasm_comgr.ps1 -TextBin text.bin -SymFile syms.txt      -Isa amdgcn-amd-amdhsa--gfx1201 -OutFile dis_gfx1201.txt

.NOTES
  Output is consumed by dlssnr_isa.py.
  If amd_comgr_3.dll is absent on your driver, change DLL in the C# block to
  amd_comgr_2.dll.
#>
param(
  [Parameter(Mandatory=$true)][string]$TextBin,
  [Parameter(Mandatory=$true)][string]$SymFile,
  [string]$Isa     = "amdgcn-amd-amdhsa--gfx1201",
  [string]$OutFile = "disassembly.txt"
)
$ErrorActionPreference = "Stop"

$src = @'
using System;
using System.IO;
using System.Text;
using System.Runtime.InteropServices;
using System.Collections.Generic;

public static class Comgr
{
    const string DLL = "amd_comgr_3.dll";
    [StructLayout(LayoutKind.Sequential)]
    public struct DisasmInfo { public ulong handle; }

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate ulong ReadMemCb(ulong from, IntPtr to, ulong size, IntPtr userData);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void PrintInsnCb(IntPtr instruction, IntPtr userData);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void PrintAnnoCb(ulong address, IntPtr userData);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int amd_comgr_create_disassembly_info(
        [MarshalAs(UnmanagedType.LPStr)] string isaName,
        ReadMemCb readMem, PrintInsnCb printInsn, PrintAnnoCb printAnno,
        out DisasmInfo info);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int amd_comgr_disassemble_instruction(
        DisasmInfo info, ulong address, IntPtr userData, out ulong size);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int amd_comgr_destroy_disassembly_info(DisasmInfo info);

    static byte[] _code;
    static List<string> _out;

    static ulong ReadMem(ulong from, IntPtr to, ulong size, IntPtr ud)
    {
        if (from >= (ulong)_code.LongLength) return 0;
        ulong avail = (ulong)_code.LongLength - from;
        ulong n = size < avail ? size : avail;
        Marshal.Copy(_code, (int)from, to, (int)n);
        return n;
    }
    static void PrintInsn(IntPtr s, IntPtr ud)
    {
        _out.Add(Marshal.PtrToStringAnsi(s));
    }
    static void PrintAnno(ulong a, IntPtr ud) { }

    public static string[] Run(string isa, byte[] code, long start, long length)
    {
        _code = code;
        _out = new List<string>();
        DisasmInfo info;
        ReadMemCb r = ReadMem; PrintInsnCb pi = PrintInsn; PrintAnnoCb pa = PrintAnno;
        int st = amd_comgr_create_disassembly_info(isa, r, pi, pa, out info);
        if (st != 0) throw new Exception("create_disassembly_info failed status=" + st);
        long addr = start; long end = start + length;
        int fails = 0;
        while (addr < end)
        {
            ulong sz;
            int s2 = amd_comgr_disassemble_instruction(info, (ulong)addr, IntPtr.Zero, out sz);
            if (s2 != 0 || sz == 0) { fails++; addr += 4; if (fails > 20000) break; continue; }
            addr += (long)sz;
        }
        amd_comgr_destroy_disassembly_info(info);
        GC.KeepAlive(r); GC.KeepAlive(pi); GC.KeepAlive(pa);
        return _out.ToArray();
    }
}
'@

Add-Type -TypeDefinition $src -Language CSharp

$code = [System.IO.File]::ReadAllBytes($TextBin)
Write-Output "loaded $($code.Length) bytes of .text; isa=$Isa"

$syms = @()
foreach ($line in Get-Content $SymFile) {
    $p = $line -split ' ', 3
    if ($p.Count -ge 3) { $syms += [pscustomobject]@{ Off = [long]$p[0]; Size = [long]$p[1]; Name = $p[2] } }
}
Write-Output "symbols: $($syms.Count)"

$sw = New-Object System.IO.StreamWriter($OutFile, $false, [System.Text.Encoding]::UTF8)
foreach ($s in $syms) {
    $sw.WriteLine("=== KERNEL $($s.Name) off=$($s.Off) size=$($s.Size) ===")
    try {
        $ins = [Comgr]::Run($Isa, $code, $s.Off, $s.Size)
        foreach ($i in $ins) { $sw.WriteLine($i) }
        Write-Output "$($s.Name): $($ins.Count) instructions"
    } catch {
        $sw.WriteLine("ERROR: $($_.Exception.Message)")
        Write-Output "$($s.Name): ERROR $($_.Exception.Message)"
    }
}
$sw.Close()
Write-Output "WROTE $OutFile"
