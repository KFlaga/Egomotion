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

    public class STAR : IAlgorithmCreator
    {
        public List<Parameter> Parameters { get; } = new List<Parameter>()
        {
            new Parameter("maxSize", typeof(int), 45),
            new Parameter("responseTreshold", typeof(int), 20),
            new Parameter("lineTreshlodProjected", typeof(int), 10),
            new Parameter("lineThresholdBinarized", typeof(int), 31),
            new Parameter("suppressNonmaxSize", typeof(int), 5),
        };

        public object Create(List<object> values)
        {
            return Activator.CreateInstance(typeof(Emgu.CV.XFeatures2D.StarDetector), values.ToArray());
        }
    }

    public class MSD : IAlgorithmCreator
    {
        public List<Parameter> Parameters { get; } = new List<Parameter>()
        {
            new Parameter("m_patch_radius", typeof(int), 3),
            new Parameter("m_search_area_radius", typeof(int), 5),
            new Parameter("m_nms_radius", typeof(int), 5),
            new Parameter("m_nms_scale_radius", typeof(int), 0),
            new Parameter("m_th_saliency", typeof(float), 250.0f),
            new Parameter("m_kNN", typeof(int), 4),
            new Parameter("m_scale_factor", typeof(float), 1.25f),
            new Parameter("m_n_scales", typeof(int), -1),
            new Parameter("m_compute_orientation", typeof(bool), false),
        };

        public object Create(List<object> values)
        {
            return Activator.CreateInstance(typeof(Emgu.CV.XFeatures2D.MSDDetector), values.ToArray());
        }
    }

    public class GFTT : IAlgorithmCreator
    {
        public List<Parameter> Parameters { get; } = new List<Parameter>()
        {
            new Parameter("maxCorners", typeof(int), 1000),
            new Parameter("qualityLevel", typeof(double), 0.01),
            new Parameter("minDistance", typeof(double), 1.0),
            new Parameter("blockSize", typeof(int), 3),
            new Parameter("useHarrisDetector", typeof(bool), false),
            new Parameter("k", typeof(double), 0.04),
        };

        public object Create(List<object> values)
        {
            return Activator.CreateInstance(typeof(Emgu.CV.Features2D.GFTTDetector), values.ToArray());
        }
    }

    public class BRIEF : IAlgorithmCreator
    {
        public List<Parameter> Parameters { get; } = new List<Parameter>()
        {
            new Parameter("descriptorSize", typeof(int), 32),
        };

        public object Create(List<object> values)
        {
            return Activator.CreateInstance(typeof(Emgu.CV.XFeatures2D.BriefDescriptorExtractor), values.ToArray());
        }
    }

    public class BOOST : IAlgorithmCreator
    {
        public List<Parameter> Parameters { get; } = new List<Parameter>()
        {
            new Parameter("descriptor type", typeof(Emgu.CV.XFeatures2D.BoostDesc.DescriptorType), Emgu.CV.XFeatures2D.BoostDesc.DescriptorType.Binboost256),
            new Parameter("use scale orientation", typeof(bool), true),
            new Parameter("scale factor", typeof(float), 5.0f),
        };

        public object Create(List<object> values)
        {
            return Activator.CreateInstance(typeof(Emgu.CV.XFeatures2D.BoostDesc), values.ToArray());
        }
    }

    public class LUCID : IAlgorithmCreator
    {
        public List<Parameter> Parameters { get; } = new List<Parameter>()
        {
            new Parameter("lucid kernel", typeof(int), 2),
            new Parameter("blur kernel", typeof(int), 3),
        };

        public object Create(List<object> values)
        {
            return Activator.CreateInstance(typeof(Emgu.CV.XFeatures2D.LUCID), values.ToArray());
        }
    }

    public class PickFeatureDetector : IAlgorithmPicker
    {
        public Dictionary<string, IAlgorithmCreator> Algorithms { get; } = new Dictionary<string, IAlgorithmCreator>()
        {
            { "FAST", new FAST() },
            { "STAR", new STAR() },
            { "MSD", new MSD() },
            { "ORB", new ORB() },
            { "GFTT", new GFTT() },
        };
    }

    public class PickFeatureDescriptor : IAlgorithmPicker
    {
        public Dictionary<string, IAlgorithmCreator> Algorithms { get; } = new Dictionary<string, IAlgorithmCreator>()
        {
            { "BRIEF", new BRIEF() },
            { "LUCID", new LUCID() },
            { "BOOST", new BOOST() },
        };
    }

    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();

            parametersInput.Parameters = new List<Parameter>()
            {
                new Parameter("Take % of best features", typeof(float), 100.0f),
                new Parameter("Max image pairs for K", typeof(float), 50),
                new Parameter("Video Step", typeof(int), 1),
                new Parameter("DistanceType", typeof(DistanceType), DistanceType.Hamming),
                new Parameter("Feature Detector", typeof(PickFeatureDetector), null),
                new Parameter("Feature Descriptor", typeof(PickFeatureDescriptor), null),
            };
        }

        Feature2D Detector => (Feature2D)Parameter.ValueFor("Feature Detector", parametersInput.Parameters, parametersInput.Values);
        Feature2D Descriptor => (Feature2D)Parameter.ValueFor("Feature Descriptor", parametersInput.Parameters, parametersInput.Values);
        double TakeBest => ((float)Parameter.ValueFor("Take % of best features", parametersInput.Parameters, parametersInput.Values)) / 100.0;
        int MaxPairsForK => ((int)Parameter.ValueFor("Max image pairs for K", parametersInput.Parameters, parametersInput.Values));
        int Step => ((int)Parameter.ValueFor("Video Step", parametersInput.Parameters, parametersInput.Values));
        DistanceType DistanceType => ((DistanceType)Parameter.ValueFor("DistanceType", parametersInput.Parameters, parametersInput.Values));
        Image<Arthmetic, Double> matK;

        private void ProcessImage(object sender, RoutedEventArgs e)
        {
            if (leftView.loadedImage == null || middleView.loadedImage == null)
            {
                MessageBox.Show("Image needs to be loaded first");
                return;
            }
            
            MatchingWindow matchingWindow = new MatchingWindow();
            matchingWindow.Show();
            matchingWindow.ProcessImages(leftView.loadedImage.Mat, middleView.loadedImage.Mat, Detector, Descriptor, DistanceType, TakeBest);
        }

        private void ProcessImageTriplet(object sender, RoutedEventArgs e)
        {
            if (leftView.loadedImage == null || middleView.loadedImage == null || rightView.loadedImage == null)
            {
                MessageBox.Show("Image needs to be loaded first");
                return;
            }

            TripletMatchingWindow matchingWindow = new TripletMatchingWindow();
            matchingWindow.Show();
            matchingWindow.ProcessImages(leftView.loadedImage.Mat, middleView.loadedImage.Mat, rightView.loadedImage.Mat, Detector, Descriptor, DistanceType);
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
                }
                catch(Exception ex)
                {
                    dataset = null;
                    MessageBox.Show(string.Format("{0} is not a valid dataset. {1}", dir, ex.Message));
                    return;
                }
            });

            if(dataset != null)
            {
                player.K = matK;
                player.Detector = Detector;
                player.Descriptor = Descriptor;
                player.DistanceType = DistanceType;
                player.TakeBest = TakeBest;
                player.Step = Step;
                player.Frames = dataset.Frames;
            }
        }

        private void LoadDataset3(object sender, RoutedEventArgs e)
        {
            FileOp.OpenFolder((dir) =>
            {
                try
                {
                    dataset = Dataset.Load(dir, datasetInterval);
                }
                catch (Exception ex)
                {
                    dataset = null;
                    MessageBox.Show(string.Format("{0} is not a valid dataset. {1}", dir, ex.Message));
                    return;
                }
            });

            if (dataset != null)
            {
                player3.Detector = Detector;
                player3.Descriptor = Descriptor;
                player3.DistanceType = DistanceType;
                player3.TakeBest = TakeBest;
                player3.Step = Step;
                player3.Frames = dataset.Frames;
            }
        }

        private void LoadParK(object sender, RoutedEventArgs e)
        {
            FileOp.LoadFromFile((file, path) => {
                SaveAndLoad.LoadCalibration(file, out Mat camMat, out VectorOfFloat distCoeffs);
                matK = camMat.ToImage<Arthmetic, Double>();
            });
        }
        private void LoadVideoFromFile(object sender, RoutedEventArgs e)
        {
            if(matK == null)
            {
                MessageBox.Show("Need to load calibration first");
                return;
            }

            List<Mat> framesFromVideo = ImageLoader.LoadVideo();
            myplayer.K = matK;
            myplayer.Detector = Detector;
            myplayer.Descriptor = Descriptor;
            myplayer.DistanceType = DistanceType;
            myplayer.TakeBest = TakeBest;
            myplayer.Step = Step;
            myplayer.Frames = framesFromVideo;
        }
    }
}
