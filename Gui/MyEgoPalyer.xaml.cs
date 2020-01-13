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
    public partial class MyEgoPlayer : UserControl
    {
        List<Mat> frames;
        int framesPerSecond;
        Timer nextFrameTimer = new Timer();
        int currentFrame = 0;
        public Image<Arthmetic, double> K;
        bool isRunning = false;

        Image<Arthmetic, double> totalRotation;
        Image<Arthmetic, double> totalTranslation;

        public event EventHandler Reset;

        public List<Mat> Frames
        {
            get { return frames; }
            set
            {
                frames = value;
                nextFrameTimer?.Stop();

                totalRotation = new Image<Arthmetic, double>(3, 3);
                totalRotation[0, 0] = 1;
                totalRotation[1, 1] = 1;
                totalRotation[2, 2] = 1;
                totalTranslation = new Image<Arthmetic, double>(1, 3);

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

        public MyEgoPlayer()
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
            UdpateFrame(currentFrame + Step);
        }

        bool recursive = false;

        private void UdpateFrame(int n)
        {
            if(Frames == null || n >= Frames.Count - Step || n < 0)
            {
                isRunning = false;
                nextFrameTimer.Stop();
                return;
            }

            Dispatcher.BeginInvoke((Action)(() =>
            {
                currentFrame = n;

                try
                {
                    var frame = frames[n];
                    var frame2 = frames[n + Step];

                    var mat = frame.ToImage<Bgr, byte>();
                    var mat2 = frame2.ToImage<Bgr, byte>();

                    double maxDistance = Math.Max(10.0, 0.05 * (frame.Rows + frame.Cols));
                    OdometerFrame odometerFrame = FindTransformation.GetOdometerFrame(mat.Mat, mat2.Mat, Detector, Descriptor, DistanceType, maxDistance, K);
                    if (odometerFrame != null)
                    {
                        videoViewer.Source = ImageLoader.ImageSourceForBitmap(frame.Bitmap); 
                        recursive = true;
                        frameProgression.Value = n;
                        recursive = false;
                        frameCurrentLabel.Content = n;

                        totalRotation = odometerFrame.RotationMatrix.Multiply(totalRotation);
                        var rotationEuler = RotationConverter.MatrixToEulerXYZ(totalRotation);
                        totalTranslation = totalTranslation + odometerFrame.Translation;

                        infoComputed.Text = FormatInfo(odometerFrame.Translation, odometerFrame.Rotation, "Comp Diff");
                        infoComputedCumulative.Text = FormatInfo(totalTranslation, rotationEuler, "Comp Cumulative");
                        infoK.Text = FormatInfoK(odometerFrame);

                        MatchDrawer.DrawFeatures(mat.Mat, mat2.Mat, odometerFrame.Match, TakeBest, matchedView);
                    }
                }
                catch(Exception e)
                {
                    infoComputed.Text = "Error!";
                }

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
        private void LoadVideoFromFile() { 
        }
    }
}
