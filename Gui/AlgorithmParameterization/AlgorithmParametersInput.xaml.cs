using System.Collections.Generic;
using System.Windows.Controls;

namespace Egomotion
{
    public partial class AlgorithmParametersInput : UserControl
    {
        private List<Parameter> parameters;

        public List<Parameter> Parameters
        {
            get { return parameters; }
            set
            {
                parameters = value;
                layout.Children.Clear();

                foreach(var p in parameters)
                {
                    if (p.IsVisible)
                    {
                        layout.Children.Add(new ParameterInput()
                        {
                            Parameter = p
                        });
                    }
                }
            }
        }

        public List<object> Values
        {
            get
            {
                List<object> values = new List<object>();
                int nextInput = 0;
                for(int i = 0; i < Parameters.Count; ++i)
                {
                    if(Parameters[i].IsVisible)
                    {
                        ParameterInput input = (ParameterInput)layout.Children[nextInput++];
                        values.Add(input.Value ?? parameters[i].Default);
                    }
                    else
                    {
                        values.Add(parameters[i].Default);
                    }
                }
                return values;
            }
        }

        public AlgorithmParametersInput()
        {
            InitializeComponent();
        }
    }
}
