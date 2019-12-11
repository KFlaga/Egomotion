using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;

namespace Egomotion
{
    public partial class MatchingWindow : Window
    {
        public MatchingWindow()
        {
            InitializeComponent();
        }

        public void ProcessImages(Mat left, Mat right, Feature2D detector)
        {
            if (left == null)
            {
                MessageBox.Show("Image needs to be loaded first");
                return;
            }

            var match = MatchImagePair.Match(left, right, detector);
            DrawFeatures(left, right, match);

            var F = ComputeMatrix.F(match.LeftPoints, match.RightPoints);
            var K = EstimateCameraFromImagePair.K(F, left.Width, right.Height);
            var E = ComputeMatrix.E(F, K);
            FindTransformation.DecomposeToRT(E, out Image<Arthmetic, double> R, out Image<Arthmetic, double> t);
            PrintMatricesInfo(E, K, R, t);
        }

        private void DrawFeatures(Mat left, Mat right, MacthingResult match)
        {
            Mat matchesImage = new Mat();
            VectorOfVectorOfDMatch matches2 = new VectorOfVectorOfDMatch();
            VectorOfKeyPoint vectorOfKp1 = new VectorOfKeyPoint(match.LeftKps);
            VectorOfKeyPoint vectorOfKp2 = new VectorOfKeyPoint(match.RightKps);
            matches2.Push(match.Matches);
            Features2DToolbox.DrawMatches(left, vectorOfKp1, right, vectorOfKp2, matches2, matchesImage, new Bgr(Color.Red).MCvScalar, new Bgr(Color.Blue).MCvScalar);

            macthedView.Source = ImageLoader.ImageSourceForBitmap(matchesImage.Bitmap);

            DrawCricles(leftView, left, match.LeftKps);
            DrawCricles(rightView, right, match.RightKps);
        }

        private void DrawCricles(ImageViewer view, Mat image, MKeyPoint[] points)
        {
            var processedImage = image.Clone();
            foreach (var kp in points)
            {
                DrawCricle(processedImage, new Bgr(Color.Wheat), new System.Drawing.Point((int)kp.Point.X, (int)kp.Point.Y), new System.Drawing.Size(10, 10));
            }
            view.Source = ImageLoader.ImageSourceForBitmap(processedImage.Bitmap);
        }

        private void DrawCricle(Mat image, Bgr color, System.Drawing.Point center, System.Drawing.Size size)
        {
            RotatedRect rect = new RotatedRect(center, size, 0);
            CvInvoke.Ellipse(image, rect, color.MCvScalar, 1);
        }

        private void PrintMatricesInfo(Image<Arthmetic, double> E, Image<Arthmetic, double> K, Image<Arthmetic, double> R, Image<Arthmetic, double> T)
        {
            StringBuilder sb = new StringBuilder();

            Svd svd = new Svd(E);
            sb.AppendLine(string.Format("E s1 = {0}, s2 = {1}", svd.S[0, 0], svd.S[1, 0]));

            sb.AppendLine();
            sb.AppendLine(string.Format("fx = {0}, fy = {1}", K[0, 0], K[1, 1]));
            sb.AppendLine(string.Format("px = {0}, py = {1}", K[0, 2], K[1, 2]));

            var t = ComputeMatrix.CrossProductToVector(T);
            sb.AppendLine();
            sb.AppendLine(string.Format("tx = {0} ", t[0, 0]));
            sb.AppendLine(string.Format("ty = {0}", t[1, 0]));
            sb.AppendLine(string.Format("tz = {0},", t[2, 0]));

            var r = RotationConverter.MatrixToEulerXYZ(R);
            sb.AppendLine();
            sb.AppendLine(string.Format("rx = {0} ", r[0, 0]));
            sb.AppendLine(string.Format("ry = {0}", r[1, 0]));
            sb.AppendLine(string.Format("rz = {0},", r[2, 0]));

            info.Text = sb.ToString();
        }
    }
}
