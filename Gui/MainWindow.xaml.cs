using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Timers;
using System.Windows;
using System.Windows.Media.Imaging;

namespace Egomotion
{
    public class FAST : IAlgorithmCreator
    {
        public List<Parameter> Parameters { get; } = new List<Parameter>()
        {
            new Parameter("Threshold", typeof(int), 10),
            new Parameter("Nonmax Supression", typeof(bool), false),
            new Parameter("Type", typeof(Emgu.CV.Features2D.FastFeatureDetector.DetectorType), FastFeatureDetector.DetectorType.Type9_16)
        };

        public object Create(List<object> values)
        {
            // Works only if Parameters have same order as FastFeatureDetector's arguments
            return Activator.CreateInstance(typeof(Emgu.CV.Features2D.FastFeatureDetector), values.ToArray());
        }
    }
    
    public class ORB : IAlgorithmCreator
    {
        public List<Parameter> Parameters { get; } = new List<Parameter>()
        {
            new Parameter("numberOfFeatures", typeof(int), 500),
            new Parameter("scaleFactor", typeof(float), 1.2f),
            new Parameter("nLevels", typeof(int), 8),
            new Parameter("edgeThreshold", typeof(int), 31),
            new Parameter("firstLevel", typeof(int), 0),
            new Parameter("WTK_A", typeof(int), 2),
            new Parameter("scoreType", typeof(Emgu.CV.Features2D.ORBDetector.ScoreType), ORBDetector.ScoreType.Harris),
            new Parameter("patchSize", typeof(int), 31),
            new Parameter("fastThreshold", typeof(int), 20),
        };

        public object Create(List<object> values)
        {
            // Works only if Parameters have same order as ORBDetector's arguments
            return Activator.CreateInstance(typeof(Emgu.CV.Features2D.ORBDetector), values.ToArray());
        }
    }
    
    public class PickFeatureDetector : IAlgorithmPicker
    {
        public Dictionary<string, IAlgorithmCreator> Algorithms { get; } = new Dictionary<string, IAlgorithmCreator>()
        {
            { "FAST + BRIEF", new FAST() },
            { "ORB", new ORB() },
        };
    }

    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();

            parametersInput.Parameters = new List<Parameter>()
            {
                new Parameter("Take % of best features", typeof(float), 50.0f),
                new Parameter("Max image pairs for K", typeof(float), 50),
                new Parameter("Feature Detector", typeof(PickFeatureDetector), null),
            };
        }

        Feature2D Detector => (Feature2D)Parameter.ValueFor("Feature Detector", parametersInput.Parameters, parametersInput.Values);
        double TakeBest => ((float)Parameter.ValueFor("Take % of best features", parametersInput.Parameters, parametersInput.Values)) / 100.0;
        int MaxPairsForK => ((int)Parameter.ValueFor("Max image pairs for K", parametersInput.Parameters, parametersInput.Values));

        private void ProcessImage(object sender, RoutedEventArgs e)
        {
            if (leftView.loadedImage == null || rightView.loadedImage == null)
            {
                MessageBox.Show("Image needs to be loaded first");
                return;
            }
            
            MatchingWindow matchingWindow = new MatchingWindow();
            matchingWindow.Show();
            matchingWindow.ProcessImages(leftView.loadedImage.Mat, rightView.loadedImage.Mat, Detector, TakeBest);
        }
        
        Dataset dataset;
        TimeSpan datasetInterval = TimeSpan.FromMilliseconds(20);
        
        private void LoadDataset(object sender, RoutedEventArgs e)
        {
            FileOp.OpenFolder((dir) =>
            {
                try
                {
                    dataset = Dataset.Load(dir, datasetInterval);
                    
                    player.Detector = Detector;
                    player.TakeBest = TakeBest;
                    player.Frames = dataset.Frames;
                }
                catch(Exception ex)
                {
                    MessageBox.Show(string.Format("{0} is not a valid dataset. {1}", dir, ex.Message));
                }
            });
        }
    }
}
