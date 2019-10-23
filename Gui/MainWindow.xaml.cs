using Emgu.CV;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Windows;

namespace Egomotion
{
    // For test and usage of IAlgorithmCreator visualisation only
    public class TestAlgorithmA : IAlgorithmCreator
    {
        public List<Parameter> Parameters { get; } = new List<Parameter>()
        {
            new Parameter("Threshold", typeof(int), 10),
            new Parameter("Nonmax Supression", typeof(bool), false),
            new Parameter("Type", typeof(Emgu.CV.Features2D.FastFeatureDetector.DetectorType), Emgu.CV.Features2D.FastFeatureDetector.DetectorType.Type9_16)
        };

        public object Create(List<object> values)
        {
            // Works only if Parameters have same order as FastFeatureDetector's arguments
            return Activator.CreateInstance(typeof(Emgu.CV.Features2D.FastFeatureDetector), values.ToArray());
        }
    }

    // For test and usage of IAlgorithmCreator visualisation only
    public class TestAlgorithmB : IAlgorithmCreator
    {
        public List<Parameter> Parameters { get; } = new List<Parameter>()
        {
            new Parameter("numberOfFeatures", typeof(int), 500),
            new Parameter("scaleFactor", typeof(float), 1.2f),
            new Parameter("nLevels", typeof(int), 8),
            new Parameter("edgeThreshold", typeof(int), 31),
            new Parameter("firstLevel", typeof(int), 0),
            new Parameter("WTK_A", typeof(int), 2),
            new Parameter("scoreType", typeof(Emgu.CV.Features2D.ORBDetector.ScoreType), Emgu.CV.Features2D.ORBDetector.ScoreType.Harris),
            new Parameter("patchSize", typeof(int), 31),
            new Parameter("fastThreshold", typeof(int), 20),
        };

        public object Create(List<object> values)
        {
            // Works only if Parameters have same order as ORBDetector's arguments
            return Activator.CreateInstance(typeof(Emgu.CV.Features2D.ORBDetector), values.ToArray());
        }
    }

    // For test and usage of IAlgorithmPicker visualisation only
    public class TestAlgorithmC : IAlgorithmPicker
    {
        public Dictionary<string, IAlgorithmCreator> Algorithms { get; } = new Dictionary<string, IAlgorithmCreator>()
        {
            { "Test A", new TestAlgorithmA() },
            { "Test B", new TestAlgorithmB() },
        };
    }

    public partial class MainWindow : Window
    {
        Image<Bgr, byte> loadedImage;
        Image<Bgr, byte> processedImage;

        public MainWindow()
        {
            InitializeComponent();

            parametersInput.Parameters = new List<Parameter>()
            {
                new Parameter("Some parameter", typeof(int), 22),
                new Parameter("Feature Detector", typeof(TestAlgorithmC), null),
            };
        }

        private void LoadImage(object sender, RoutedEventArgs e)
        {
            loadedImage = ImageLoader.FromFile();
            if (loadedImage != null)
            {
                imageViewer.Source = ImageLoader.ImageSourceForBitmap(loadedImage.Bitmap);
            }
        }

        private void ProcessImage(object sender, RoutedEventArgs e)
        {
            if (loadedImage == null)
            {
                MessageBox.Show("Image needs to be loaded first");
                return;
            }

            processedImage = loadedImage.Clone();
            
            Emgu.CV.Features2D.Feature2D detector = (Emgu.CV.Features2D.Feature2D)Parameter.ValueFor("Feature Detector", parametersInput.Parameters, parametersInput.Values);

            MKeyPoint[] kps = detector.Detect(loadedImage.Mat);

            // TODO: open cv probably has some function to draw features automatically
            foreach(var kp in kps)
            {
                DrawCricle(processedImage, new Bgr(Color.Wheat), new System.Drawing.Point((int)kp.Point.X, (int)kp.Point.Y), new System.Drawing.Size(10, 10));
            }

            Emgu.CV.Util.VectorOfKeyPoint vectorOfKp = new Emgu.CV.Util.VectorOfKeyPoint(kps);

            var desc = new Emgu.CV.XFeatures2D.BriefDescriptorExtractor(32);
            Mat descriptors = new Mat();
            desc.Compute(loadedImage, vectorOfKp, descriptors);

            imageViewer.Source = ImageLoader.ImageSourceForBitmap(processedImage.Bitmap);
        }


        private void DrawCricle(Image<Bgr, byte> image, Bgr color, System.Drawing.Point center, System.Drawing.Size size)
        {
            RotatedRect rect = new RotatedRect(center, size, 0);
            CvInvoke.Ellipse(image, rect, color.MCvScalar, 1);
        }
    }
}
