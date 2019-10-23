using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Windows.Controls;

namespace Egomotion
{
    public static class ParameterInputCreator
    {
        public static ParameterValueInput CreateInput(Parameter p)
        {
            if (p.Type.IsEnum)
            {
                return CreateCombo(p);
            }
            else if (p.Type == typeof(int))
            {
                return CreateIntInput(p);
            }
            else if (p.Type == typeof(float))
            {
                return CreateFloatInput(p);
            }
            else if (p.Type == typeof(bool))
            {
                return CreateBoolInput(p);
            }
            else if (p.Type == typeof(string))
            {
                return CreateStringInput(p);
            }
            else if (p.Type.GetInterfaces().Contains(typeof(IAlgorithmPicker)))
            {
                return CreatePickAlgorithmInput(p);
            }
            else if(p.Type.GetInterfaces().Contains(typeof(IAlgorithmCreator)))
            {
                return CreateAlgorithmInput(p);
            }
            throw new ArgumentException("Unsupported parameter type " + p.Type.Name);
        }

        private static ParameterValueInput CreateStringInput(Parameter p)
        {
            return new ParameterValueInput()
            {
                Gui = new TextBox()
                {
                    Text = p.Default.ToString()
                },
                GetValue = (gui) => (gui as TextBox).Text
            };
        }

        private static ParameterValueInput CreateIntInput(Parameter p)
        {
            return new ParameterValueInput()
            {
                Gui = new TextBox()
                {
                    Text = p.Default.ToString()
                },
                GetValue = (gui) => int.Parse((gui as TextBox).Text)
            };
        }

        private static ParameterValueInput CreateFloatInput(Parameter p)
        {
            return new ParameterValueInput()
            {
                Gui = new TextBox()
                {
                    Text = p.Default.ToString()
                },
                GetValue = (gui) => float.Parse((gui as TextBox).Text)
            };
        }

        private static ParameterValueInput CreateBoolInput(Parameter p)
        {
            return new ParameterValueInput()
            {
                Gui = new CheckBox()
                {
                    IsChecked = (bool)p.Default
                },
                GetValue = (gui) => (gui as CheckBox).IsChecked.GetValueOrDefault(false)
            };
        }

        private static ParameterValueInput CreateCombo(Parameter p)
        {
            Type t = p.Type;

            ComboBox comboBox = new ComboBox();
            foreach (var name in Enum.GetNames(t))
            {
                comboBox.Items.Add(name);
            }
            comboBox.SelectedItem = p.Default.ToString();

            return new ParameterValueInput()
            {
                Gui = comboBox,
                GetValue = (gui) => Enum.Parse(t, (string)(gui as ComboBox).SelectedItem)
            };
        }

        private static ParameterValueInput CreatePickAlgorithmInput(Parameter p)
        {
            Type t = p.Type;
            IAlgorithmPicker algorithmPicker = (IAlgorithmPicker)Activator.CreateInstance(t);

            return new ParameterValueInput()
            {
                Gui = new PickAlgorithmInput()
                {
                    Algorithms = algorithmPicker.Algorithms
                },
                GetValue = (gui) => ((PickAlgorithmInput)gui).Algorithm
            };
        }

        private static ParameterValueInput CreateAlgorithmInput(Parameter p)
        {
            Type t = p.Type;
            IAlgorithmCreator algorithmCreator = (IAlgorithmCreator)Activator.CreateInstance(t);

            return new ParameterValueInput()
            {
                Gui = new AlgorithmParametersInput()
                {
                    Parameters = algorithmCreator.Parameters
                },
                GetValue = (gui) => algorithmCreator.Create(((AlgorithmParametersInput)gui).Values)
            };
        }
    }
}
