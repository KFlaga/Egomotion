using Emgu.CV;
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
    public partial class EgoPlayer : UserControl
    {
        List<DatasetFrame> frames;
        int framesPerSecond;
        Timer nextFrameTimer = new Timer();
        int currentFrame = 0;

        public List<DatasetFrame> Frames
        {
            get { return frames; }
            set
            {
                frames = value;
                nextFrameTimer?.Stop();
                rotation = frames[0].Odometry.RotationMatrix;
                
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


        public EgoPlayer()
        {
            InitializeComponent();

            nextFrameTimer = new Timer()
            {
                Interval = 20.0,
                AutoReset = false
            };
            nextFrameTimer.Elapsed += NextFrameTimer_Elapsed;
            FramesPerSecond = 50;
        }

        private void Start(object sender, RoutedEventArgs e)
        {
            if(Frames != null)
            {
                nextFrameTimer.Start();
            }
        }

        private void Stop(object sender, RoutedEventArgs e)
        {
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

        private void UdpateFrame(int n)
        {
            if(Frames == null || n >= Frames.Count - 1 || n < 0)
            {
                nextFrameTimer.Stop();
                return;
            }

            Dispatcher.BeginInvoke((Action)(() =>
            {
                currentFrame = n;
                var frame = frames[n];
                var frame2 = frames[n + 1];

                var mat = Emgu.CV.CvInvoke.Imread(frame.ImageFile, Emgu.CV.CvEnum.ImreadModes.Color).ToImage<Bgr, byte>();
                var mat2 = Emgu.CV.CvInvoke.Imread(frame2.ImageFile, Emgu.CV.CvEnum.ImreadModes.Color).ToImage<Bgr, byte>();

                var detector = new Emgu.CV.Features2D.ORBDetector();

                OdometerFrame odometerFrame = FindTransformation.GetOdometerFrame(mat.Mat, mat2.Mat, detector);
                if (odometerFrame != null)
                {

                    rotation = odometerFrame.RotationMatrix.Multiply(rotation);
                   // odometerFrame.Rotation = RotationConverter.MatrixToEulerXYZ(rotation);


                    videoViewer.Source = new BitmapImage(new Uri(frame.ImageFile, UriKind.Absolute));
                    recursive = true;
                    frameProgression.Value = n;
                    recursive = false;
                    frameCurrentLabel.Content = n;
                    infoReference.Text = FormatInfo(frame.Odometry);
                    infoComputed.Text = FormatInfo(odometerFrame);

                    StringBuilder sb = new StringBuilder();
                    sb.AppendLine(string.Format("Frame {0}", currentFrame));
                    sb.AppendLine(string.Format("X: {0}", odometerFrame.MatK[0, 0].Value.ToString("F4")));
                    sb.AppendLine(string.Format("Y: {0}", odometerFrame.MatK[1, 1].Value.ToString("F4")));
                    sb.AppendLine(string.Format("Z: {0}", odometerFrame.MatK[0, 2].Value.ToString("F4")));
                    sb.AppendLine(string.Format("Z: {0}", odometerFrame.MatK[1, 2].Value.ToString("F4")));
                    MatK.Text = sb.ToString();
                }
                nextFrameTimer.Start();

            }));
        }

        public Image<Arthmetic, double> rotation;

        private void FrameProgression_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if(!recursive)
            {
                UdpateFrame((int)e.NewValue);
            }
        }

        private double rad2deg(double rad)
        {
            return 180.0 * rad / Math.PI;
        }

        private string FormatInfo(OdometerFrame frame)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine(string.Format("Frame {0}", currentFrame));
            sb.AppendLine("Translation:");
            sb.AppendLine(string.Format("X: {0}", frame.Translation[0, 0].Value.ToString("F4")));
            sb.AppendLine(string.Format("Y: {0}", frame.Translation[1, 0].Value.ToString("F4")));
            sb.AppendLine(string.Format("Z: {0}", frame.Translation[2, 0].Value.ToString("F4")));
            sb.AppendLine("Rotation:");
            sb.AppendLine(string.Format("X: {0}", rad2deg(frame.Rotation[0, 0].Value).ToString("F4")));
            sb.AppendLine(string.Format("Y: {0}", rad2deg(frame.Rotation[1, 0].Value).ToString("F4")));
            sb.AppendLine(string.Format("Z: {0}", rad2deg(frame.Rotation[2, 0].Value).ToString("F4")));
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
