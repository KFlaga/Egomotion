using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Timers;
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
    public partial class EgoPlayer3 : UserControl
    {
        List<DatasetFrame> frames;
        int framesPerSecond;
        Timer nextFrameTimer = new Timer();
        int currentFrame = 0;
        Image<Arthmetic, double> K;
        bool isRunning = false;

        Image<Arthmetic, double> totalRotation;
        Image<Arthmetic, double> totalTranslation;

        public event EventHandler Reset;

        public List<DatasetFrame> Frames
        {
            get { return frames; }
            set
            {
                frames = value;
                nextFrameTimer?.Stop();

                totalRotation = frames[0].Odometry.RotationMatrix;
                totalTranslation = new Image<Arthmetic, double>(1,3);

                ComputeK(frames);
                Dispatcher.BeginInvoke((Action)(() =>
                {
                    frameProgression.Minimum = 0;
                    frameProgression.Maximum = frames.Count;
                    frameCountLabel.Content = frames.Count;
                }));

                UdpateFrame(0);
            }
        }
        public int FramesPerSecond
        {
            get { return framesPerSecond; }
            set
            {
                framesPerSecond = value;
                nextFrameTimer.Interval = 1000.0 / framesPerSecond;
                Dispatcher.BeginInvoke((Action)(() =>
                {
                    recursive = true;
                    framePerSecSlider.Value = framesPerSecond;
                    recursive = false;
                    framePerSecLabel.Content = framesPerSecond;
                }));
            }
        }

        public Feature2D Detector { get; set; } = new FastFeatureDetector(20, true);
        public Feature2D Descriptor { get; set; } = new Emgu.CV.XFeatures2D.BriefDescriptorExtractor(32);
        public DistanceType DistanceType { get; set; } = DistanceType.Hamming;
        public double TakeBest { get; set; } = 50.0;
        public int MaxPairsForK { get; set; } = 50;
        public int Step { get; set; } = 4;

        public void ComputeK(List<DatasetFrame> fr)
        {
            Random rand = new Random();
            int countFrame = Math.Min(MaxPairsForK, (int)Math.Ceiling(fr.Count * 0.1));

            List < Mat > checkedFrames = new List<Mat>();

            for (int c = 0; c< countFrame; c++)
            {
                int f = rand.Next(0, fr.Count - 1);
                checkedFrames.Add(CvInvoke.Imread(fr[f].ImageFile, Emgu.CV.CvEnum.ImreadModes.Color).ToImage<Bgr, byte>().Mat);
                checkedFrames.Add(CvInvoke.Imread(fr[f+1].ImageFile, Emgu.CV.CvEnum.ImreadModes.Color).ToImage<Bgr, byte>().Mat);
            }

            double maxDistance = 20.0;
            K = EstimateCameraFromImageSequence.K(checkedFrames, Detector, Descriptor, DistanceType, maxDistance);
        }

        public EgoPlayer3()
        {
            InitializeComponent();

            nextFrameTimer = new Timer()
            {
                Interval = 200.0,
                AutoReset = false
            };
            nextFrameTimer.Elapsed += NextFrameTimer_Elapsed;
            FramesPerSecond = 5;
        }

        private void Start(object sender, RoutedEventArgs e)
        {
            if(Frames != null)
            {
                isRunning = true;
                nextFrameTimer.Start();
            }
        }
        
        private void Stop(object sender, RoutedEventArgs e)
        {
            isRunning = false;
            nextFrameTimer.Stop();
        }

        private void SwitchOverlay(object sender, RoutedEventArgs e)
        {
            if(overlayInfo.Visibility == Visibility.Visible)
            {
                overlayInfo.Visibility = Visibility.Hidden;
            }
            else
            {
                overlayInfo.Visibility = Visibility.Visible;
            }
        }

        private void NextFrameTimer_Elapsed(object sender, ElapsedEventArgs e)
        {
            UdpateFrame(currentFrame + 1);
        }

        bool recursive = false;

        double lastScale = 1.0;

        private void UdpateFrame(int n)
        {
            if(Frames == null || n >= Frames.Count - 2 || n < 0)
            {
                isRunning = false;
                nextFrameTimer.Stop();
                return;
            }

            Dispatcher.BeginInvoke((Action)(() =>
            {
                currentFrame = n;
                var frame = frames[n];
                var frame2 = frames[n + 1];
                var frame3 = frames[n + 2];

                var mat = CvInvoke.Imread(frame.ImageFile, Emgu.CV.CvEnum.ImreadModes.Color).ToImage<Bgr, byte>();
                var mat2 = CvInvoke.Imread(frame2.ImageFile, Emgu.CV.CvEnum.ImreadModes.Color).ToImage<Bgr, byte>();
                var mat3 = CvInvoke.Imread(frame2.ImageFile, Emgu.CV.CvEnum.ImreadModes.Color).ToImage<Bgr, byte>();

           //     try
           //     {
                    double maxDistance = 20.0;
                    OdometerFrame odometerFrame = ThreeViews.GetOdometerFrame3(mat.Mat, mat2.Mat, mat3.Mat, 
                        lastScale, out double thisScale,
                        Detector, Descriptor, DistanceType, maxDistance, K);
                    if (odometerFrame != null)
                    {
                        videoViewer.Source = new BitmapImage(new Uri(frame.ImageFile, UriKind.Absolute));
                        recursive = true;
                        frameProgression.Value = n;
                        recursive = false;
                        frameCurrentLabel.Content = n;

                        totalRotation = odometerFrame.RotationMatrix.Multiply(totalRotation);
                        var rotationEuler = RotationConverter.MatrixToEulerXYZ(totalRotation);
                        totalTranslation = totalTranslation + odometerFrame.Translation;

                        var refTranslation = frame2.Odometry.Translation.Sub(frames[0].Odometry.Translation);
                        var refRotation = frames[0].Odometry.RotationMatrix.T().Multiply(frame2.Odometry.RotationMatrix);
                        var refRotationEuler = RotationConverter.MatrixToEulerXYZ(refRotation);

                        var refTranslationDiff = frame2.Odometry.Translation.Sub(frame.Odometry.Translation);
                        var refRotationDiff = frame.Odometry.RotationMatrix.T().Multiply(frame2.Odometry.RotationMatrix);
                        var refRotationDiffEuler = RotationConverter.MatrixToEulerXYZ(refRotationDiff);

                        infoReference.Text = FormatInfo(refTranslation, refRotationEuler, "Ref Cumulative");
                        infoReferenceDiff.Text = FormatInfo(refTranslationDiff, refRotationDiffEuler, "Ref Diff");
                        infoComputed.Text = FormatInfo(odometerFrame.Center, odometerFrame.Rotation, "Comp Diff");
                        infoComputedCumulative.Text = FormatInfo(totalTranslation, rotationEuler, "Comp Cumulative");
                        infoK.Text = FormatInfoK(odometerFrame);

                        MatchDrawer.DrawFeatures(mat.Mat, mat2.Mat, odometerFrame.Match, TakeBest, matchedView);

                        lastScale = thisScale * lastScale;
                    }
           //     }
           //     catch(Exception e)
          //      {
          //          infoComputed.Text = "Error!";
           //     }

                if(isRunning)
                    nextFrameTimer.Start();
            }));
        }


        private void FrameProgression_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if(!recursive)
            {
                UdpateFrame((int)e.NewValue);
            }
        }

        private string FormatInfoK(OdometerFrame frame)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine(string.Format("fx: {0}", frame.MatK[0, 0].Value.ToString("F4")));
            sb.AppendLine(string.Format("fy: {0}", frame.MatK[1, 1].Value.ToString("F4")));
            sb.AppendLine(string.Format("px: {0}", frame.MatK[0, 2].Value.ToString("F4")));
            sb.AppendLine(string.Format("py: {0}", frame.MatK[1, 2].Value.ToString("F4")));
            infoK.Text = sb.ToString();
            return sb.ToString();

        }

        private double rad2deg(double rad)
        {
            return 180.0 * rad / Math.PI;
        }

        private string FormatInfo(Image<Arthmetic, double> translation, Image<Arthmetic, double> roatation, string name)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine(name);
            sb.AppendLine(string.Format("Frame {0}", currentFrame));
            sb.AppendLine("Translation:");
            sb.AppendLine(string.Format("X: {0}", (translation[0, 0]).Value.ToString("F4")));
            sb.AppendLine(string.Format("Y: {0}", (translation[1, 0]).Value.ToString("F4")));
            sb.AppendLine(string.Format("Z: {0}", (translation[2, 0]).Value.ToString("F4")));
            sb.AppendLine("Rotation:");
            sb.AppendLine(string.Format("X: {0}", rad2deg((roatation[0, 0])).ToString("F4")));
            sb.AppendLine(string.Format("Y: {0}", rad2deg((roatation[1, 0])).ToString("F4")));
            sb.AppendLine(string.Format("Z: {0}", rad2deg((roatation[2, 0])).ToString("F4")));
            return sb.ToString();
        }

        private void FramePerSecSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (!recursive)
            {
                FramesPerSecond = (int)e.NewValue;
            }
        }
    }
}
