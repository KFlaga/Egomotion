using System;
using System.Collections.Generic;

namespace Egomotion
{
    public class Parameter
    {
        public string Name { get; private set; }
        public Type Type { get; private set; }
        public object Default { get; private set; }
        public bool IsVisible { get; private set; }

        public Parameter(string name, Type type, object def, bool isVisible = true)
        {
            Name = name;
            Type = type;
            Default = def;
            IsVisible = isVisible;
        }

        public static object ValueFor(string name, List<Parameter> parameters, List<object> values)
        {
            int index = parameters.FindIndex((p) => p.Name == name);
            return values[index];
        }
    }

    public interface IAlgorithmCreator
    {
        List<Parameter> Parameters { get; }
        object Create(List<object> values);
    }
    
    public interface IAlgorithmPicker
    {
        Dictionary<string, IAlgorithmCreator> Algorithms { get; }
    }
}
