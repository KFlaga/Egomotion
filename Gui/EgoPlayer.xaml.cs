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
    public partial class EgoPlayer : UserControl
    {
        List<DatasetFrame> frames;
        int framesPerSecond;
        Timer nextFrameTimer = new Timer();
        int currentFrame = 0;
        Image<Arthmetic, double> K;
        bool isRunning = false;

        public List<DatasetFrame> Frames
        {
            get { return frames; }
            set
            {
                frames = value;
                nextFrameTimer?.Stop();

                rotation = frames[0].Odometry.RotationMatrix;
                translation = new Image<Arthmetic, double>(1,3);

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

        public Feature2D Detector { get; set; }

        public void ComputeK(List<DatasetFrame> fr)
        {
            Random rand = new Random();
            int countFrame = Math.Min(50, (int)Math.Ceiling(fr.Count * 0.05));

            List < Mat > checkedFrames = new List<Mat>();

            for (int c = 0; c< countFrame; c++)
            {
                int f = rand.Next(0, fr.Count - 1);
                checkedFrames.Add(CvInvoke.Imread(fr[f].ImageFile, Emgu.CV.CvEnum.ImreadModes.Color).ToImage<Bgr, byte>().Mat);
                checkedFrames.Add(CvInvoke.Imread(fr[f+1].ImageFile, Emgu.CV.CvEnum.ImreadModes.Color).ToImage<Bgr, byte>().Mat);
            }

            K = EstimateCameraFromImageSequence.K(checkedFrames, Detector);
        }

        public EgoPlayer()
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

        private void UdpateFrame(int n)
        {
            if(Frames == null || n >= Frames.Count - 1 || n < 0)
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

                var mat = CvInvoke.Imread(frame.ImageFile, Emgu.CV.CvEnum.ImreadModes.Color).ToImage<Bgr, byte>();
                var mat2 = CvInvoke.Imread(frame2.ImageFile, Emgu.CV.CvEnum.ImreadModes.Color).ToImage<Bgr, byte>();
                
                OdometerFrame odometerFrame = FindTransformation.GetOdometerFrame(mat.Mat, mat2.Mat, Detector, K);
                if (odometerFrame != null)
                {

                    rotation = odometerFrame.RotationMatrix.Multiply(rotation);
                    odometerFrame.Rotation = RotationConverter.MatrixToEulerXYZ(rotation);

                    translation = translation + odometerFrame.Translation;
                    odometerFrame.Translation = translation;
                    
                    videoViewer.Source = new BitmapImage(new Uri(frame.ImageFile, UriKind.Absolute));
                    recursive = true;
                    frameProgression.Value = n;
                    recursive = false;
                    frameCurrentLabel.Content = n;
                    infoReference.Text = FormatInfo(frame.Odometry, frames[0].Odometry);
                    OdometerFrame zeros = new OdometerFrame();
                    zeros.Translation = new Image<Arthmetic, double>(1, 3);
                    zeros.Rotation = new Image<Arthmetic, double>(1, 3);
                    
                    infoComputed.Text = FormatInfo(odometerFrame, zeros);

                    StringBuilder sb = new StringBuilder();
                    sb.AppendLine(string.Format("Frame {0}", currentFrame));
                    sb.AppendLine(string.Format("X: {0}", odometerFrame.MatK[0, 0].Value.ToString("F4")));
                    sb.AppendLine(string.Format("Y: {0}", odometerFrame.MatK[1, 1].Value.ToString("F4")));
                    sb.AppendLine(string.Format("Z: {0}", odometerFrame.MatK[0, 2].Value.ToString("F4")));
                    sb.AppendLine(string.Format("Z: {0}", odometerFrame.MatK[1, 2].Value.ToString("F4")));
                    MatK.Text = sb.ToString();
                }

                if(isRunning)
                    nextFrameTimer.Start();
            }));
        }

        public Image<Arthmetic, double> rotation;
        public Image<Arthmetic, double> translation;


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

        private string FormatInfo(OdometerFrame frame, OdometerFrame first)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine(string.Format("Frame {0}", currentFrame));
            sb.AppendLine("Translation:");
            sb.AppendLine(string.Format("X: {0}", (frame.Translation[0, 0] - first.Translation[0,0]).ToString("F4")));
            sb.AppendLine(string.Format("Y: {0}", (frame.Translation[1, 0] - first.Translation[1, 0]).ToString("F4")));
            sb.AppendLine(string.Format("Z: {0}", (frame.Translation[2, 0] - first.Translation[2, 0]).ToString("F4")));
            sb.AppendLine("Rotation:");
            sb.AppendLine(string.Format("X: {0}", rad2deg((frame.Rotation[0, 0] )).ToString("F4")));
            sb.AppendLine(string.Format("Y: {0}", rad2deg((frame.Rotation[1, 0] )).ToString("F4")));
            sb.AppendLine(string.Format("Z: {0}", rad2deg((frame.Rotation[2, 0] )).ToString("F4")));
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
