using System.Diagnostics;
using System.Reflection;
using System.Reflection.Metadata;
using System.Runtime.InteropServices;


namespace CoreInterop;

public class ScenariumCore {
    public static void Init() {
        CoreNative.LoadDll();
    }

    public ScenariumCore() {
        if (!CoreNative.IsLoaded) {
            throw new InvalidOperationException("CoreNative is not loaded, call CoreNative.LoadDll() first.");
        }
    }


    public String Test() {
        var buf = CoreNative.test3();
        var result = buf.ToString();
        // buf.Dispose();
        return result;
    }
}