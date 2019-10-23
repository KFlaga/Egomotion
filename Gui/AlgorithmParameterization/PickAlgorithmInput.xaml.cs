using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace Egomotion
{
    public partial class PickAlgorithmInput : UserControl
    {
        Dictionary<string, IAlgorithmCreator> algorithms;
        public Dictionary<string, IAlgorithmCreator> Algorithms
        {
            get { return algorithms; }
            set
            {
                algorithms = value;
                algorithmsCombo.Items.Clear();

                foreach(var a in algorithms)
                {
                    algorithmsCombo.Items.Add(new ComboBoxItem()
                    {
                        Content = a.Key,
                        Tag = a.Value
                    });
                }

                if(algorithms.Count > 0)
                {
                    algorithmsCombo.SelectedIndex = 0;
                }
            }
        }

        Func<FrameworkElement, object> getAlgorithm;
        public object Algorithm
        {
            get
            {
                return getAlgorithm?.Invoke((FrameworkElement)layout.Children[1]);
            }
        }

        public PickAlgorithmInput()
        {
            InitializeComponent();
        }

        private void AlgorithmsCombo_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if(e.AddedItems.Count > 0)
            {
                ComboBoxItem item = (ComboBoxItem)e.AddedItems[0];
                IAlgorithmCreator algorithm = (IAlgorithmCreator)item.Tag;
                
                if(layout.Children.Count > 1)
                {
                    layout.Children.RemoveAt(1);
                }
                var input = ParameterInputCreator.CreateInput(new Parameter("Algorithm", item.Tag.GetType(), null));
                layout.Children.Add(input.Gui);
                getAlgorithm = input.GetValue;
            }
        }
    }
}
