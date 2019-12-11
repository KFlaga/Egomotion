using Emgu.CV;
using Emgu.CV.Structure;
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
        public MainWindow()
        {
            InitializeComponent();

            parametersInput.Parameters = new List<Parameter>()
            {
                new Parameter("Some parameter", typeof(int), 22),
                new Parameter("Feature Detector", typeof(TestAlgorithmC), null),
            };
        }

        private void ProcessImage(object sender, RoutedEventArgs e)
        {
            if (imageLoad.loadedImage == null)
            {
                MessageBox.Show("Image needs to be loaded first");
                return;
            }

            ProcessSIngleImage(imageLoad, out MKeyPoint[] kps1, out Mat desc1);
            ProcessSIngleImage(imageLoad2, out MKeyPoint[] kps2, out Mat desc2);

            Emgu.CV.Features2D.BFMatcher m = new Emgu.CV.Features2D.BFMatcher(Emgu.CV.Features2D.DistanceType.L2);
            Emgu.CV.Util.VectorOfDMatch matches = new Emgu.CV.Util.VectorOfDMatch();
            m.Match(desc1, desc2, matches);

            Mat result = new Mat();
            Emgu.CV.Util.VectorOfKeyPoint vectorOfKp1 = new Emgu.CV.Util.VectorOfKeyPoint(kps1);
            Emgu.CV.Util.VectorOfKeyPoint vectorOfKp2 = new Emgu.CV.Util.VectorOfKeyPoint(kps2);
            Emgu.CV.Util.VectorOfVectorOfDMatch matches2 = new Emgu.CV.Util.VectorOfVectorOfDMatch();
            matches2.Push(matches);
            Emgu.CV.Features2D.Features2DToolbox.DrawMatches(imageLoad.loadedImage, vectorOfKp1, imageLoad2.loadedImage, vectorOfKp2, matches2, result, new Bgr(Color.Red).MCvScalar, new Bgr(Color.Blue).MCvScalar);
        
            imageLoad2.Source = ImageLoader.ImageSourceForBitmap(result.Bitmap);

            FindTransformation.FindMatches(matches, kps1, kps2, out Emgu.CV.Util.VectorOfPointF point1, out Emgu.CV.Util.VectorOfPointF point2);
            var F = FindTransformation.ComputeFundametnalMatrix(point1, point2);
            var K = FindTransformation.Optimal(F.Mat, imageLoad.loadedImage.Width, imageLoad.loadedImage.Height);
            var E = K.T().Multiply(F).Multiply(K);
            FindTransformation.ReturnRT(E, out Image<Arthmetic, double> R, out Image<Arthmetic, double> t);
        }

        private void ProcessSIngleImage(ImageLoad il, out MKeyPoint[] kps, out Mat descriptors)
        {
            Emgu.CV.Features2D.Feature2D detector = (Emgu.CV.Features2D.Feature2D)Parameter.ValueFor("Feature Detector", parametersInput.Parameters, parametersInput.Values);

            FindTransformation.FindFeatures(il.loadedImage.Mat, detector, out kps, out descriptors);
            var processedImage = il.loadedImage.Clone();

            // TODO: open cv probably has some function to draw features automatically
            foreach (var kp in kps)
            {
                DrawCricle(processedImage, new Bgr(Color.Wheat), new System.Drawing.Point((int)kp.Point.X, (int)kp.Point.Y), new System.Drawing.Size(10, 10));
            }
            il.Source = ImageLoader.ImageSourceForBitmap(processedImage.Bitmap);
        }

        private void DrawCricle(Image<Bgr, byte> image, Bgr color, System.Drawing.Point center, System.Drawing.Size size)
        {
            RotatedRect rect = new RotatedRect(center, size, 0);
            CvInvoke.Ellipse(image, rect, color.MCvScalar, 1);
        }
        
        Dataset dataset;
        Timer playTimer;
        int currentFrame = 0;
        TimeSpan datasetInterval = TimeSpan.FromMilliseconds(20);
        
        private void LoadDataset(object sender, RoutedEventArgs e)
        {
            FileOp.OpenFolder((dir) =>
            {
                try
                {
                    dataset = Dataset.Load(dir, datasetInterval);
                    player.Frames = dataset.Frames;
                }
                catch(Exception ex)
                {
                    MessageBox.Show(string.Format("{0} is not a valid dataset. {1}", dir, ex.Message));
                }
            });
        }

        private void imageLoad_Loaded(object sender, RoutedEventArgs e)
        {

        }
    }
}
