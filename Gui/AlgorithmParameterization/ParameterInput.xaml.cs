using System;
using System.Windows;
using System.Windows.Controls;

namespace Egomotion
{
    public class ParameterValueInput
    {
        public FrameworkElement Gui { get; set; }
        public Func<FrameworkElement, object> GetValue { get; set; }
    }

    public partial class ParameterInput : UserControl
    {
        ParameterValueInput valueInput;
        public ParameterValueInput ValueInput
        {
            get { return valueInput; }
            set
            {
                valueInput = value;
                if(layout.Children.Count > 1)
                {
                    layout.Children.RemoveAt(1);
                }
                layout.Children.Add(valueInput.Gui);
            }
        }

        Parameter p;
        public Parameter Parameter
        {
            get { return p; }
            set
            {
                p = value;
                ValueInput = ParameterInputCreator.CreateInput(p);
                nameLabel.Content = p.Name;
            }
        }

        public object Value
        {
            get
            {
                try
                {
                    return valueInput.GetValue(valueInput.Gui);
                }
                catch(Exception)
                {
                    return null;
                }
            }
        }

        public ParameterInput()
        {
            InitializeComponent();
        }
    }
}
