using System.Runtime.InteropServices;
using YamlDotNet.Core;
using YamlDotNet.Core.Events;
using YamlDotNet.Serialization;

namespace CoreInterop.Utils;

public readonly struct Uuid : IEquatable<Uuid> {
    private readonly UInt64 _a = 0;
    private readonly UInt64 _b = 0;


    public Uuid() {
    }

    public static Uuid NewV4() {
        return LibraryLoader.uuid_new_v4_extern();
    }

    public static Uuid FromString(String s) {
        using var buf = new FfiBuf(s);
        return LibraryLoader.uuid_from_string_extern(buf._buf_intern);
    }

    public override String ToString() {
        using FfiBuf buf = LibraryLoader.uuidToString(this);
        return buf.ToString();
    }

    public override bool Equals(object obj) {
        return obj is Uuid other && Equals(other);
    }

    public bool Equals(Uuid other) {
        return _a == other._a && _b == other._b;
    }

    public override int GetHashCode() {
        return HashCode.Combine(_a, _b);
    }
}

public class UuidConverter : IYamlTypeConverter {
    public bool Accepts(Type type) {
        return type == typeof(Uuid);
    }

    public object ReadYaml(IParser parser, Type type) {
        var scalar = parser.Consume<Scalar>();
        return Uuid.FromString(scalar.Value);
    }

    public void WriteYaml(IEmitter emitter, object value, Type type) {
        var uuid = (Uuid)value!;
        emitter.Emit(new Scalar(uuid.ToString()));
    }
}