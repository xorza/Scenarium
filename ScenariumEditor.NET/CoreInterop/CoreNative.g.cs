// <auto-generated>
// This code is generated by csbindgen.
// DON'T CHANGE THIS DIRECTLY.
// </auto-generated>
#pragma warning disable CS8500
#pragma warning disable CS8981
using System;
using System.Runtime.InteropServices;


namespace CoreInterop
{
    internal static unsafe partial class CoreNative
    {
        const string __DllName = "core_interop";



        [DllImport(__DllName, EntryPoint = "create_context", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void* create_context();

        [DllImport(__DllName, EntryPoint = "destroy_context", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void destroy_context(byte* ctx);

        [DllImport(__DllName, EntryPoint = "dummy", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void dummy(FfiBuf _a, FfiStr _b, FfiStrVec _c, FfiId _d);

        [DllImport(__DllName, EntryPoint = "get_nodes", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern FfiBuf get_nodes(byte* ctx);

        [DllImport(__DllName, EntryPoint = "new_node", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern FfiNode new_node(byte* ctx, FfiId func_id);

        [DllImport(__DllName, EntryPoint = "dummy1", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void dummy1(FfiNode _a, FfiInput _b);

        [DllImport(__DllName, EntryPoint = "get_funcs", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern FfiBuf get_funcs(byte* ctx);

        [DllImport(__DllName, EntryPoint = "dummy2", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void dummy2(FfiFunc _a);


    }

    [StructLayout(LayoutKind.Sequential)]
    internal unsafe partial struct FfiBuf
    {
        public byte* bytes;
        public uint length;
        public uint capacity;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal unsafe partial struct FfiId
    {
        public FfiStr Item1;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal unsafe partial struct FfiStr
    {
        public FfiBuf Item1;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal unsafe partial struct FfiStrVec
    {
        public FfiBuf Item1;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal unsafe partial struct FfiInput
    {
        public byte a;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal unsafe partial struct FfiNode
    {
        public FfiId id;
        public FfiId func_id;
        public FfiStr name;
        [MarshalAs(UnmanagedType.U1)] public bool is_output;
        [MarshalAs(UnmanagedType.U1)] public bool cache_outputs;
        public FfiBuf inputs;
        public FfiBuf events;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal unsafe partial struct FfiFunc
    {
        public FfiId id;
        public FfiStr name;
        public FfiStr category;
        public FuncBehavior behaviour;
        [MarshalAs(UnmanagedType.U1)] public bool is_output;
        public FfiBuf inputs;
        public FfiBuf outputs;
        public FfiStrVec events;
    }


    internal enum FuncBehavior : uint
    {
        Active,
        Passive,
    }


}
