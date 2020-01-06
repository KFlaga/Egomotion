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

        public void ProcessImages(Mat left, Mat right, Feature2D detector, Feature2D descriptor, DistanceType distanceType, double takeBest)
        {
            var match = MatchImagePair.Match(left, right, detector, descriptor, distanceType, 20.0);
            DrawFeatures(left, right, match, takeBest);

            var lps = match.LeftPoints.ToArray().Take((int)(match.LeftPoints.Size * takeBest)).ToArray();
            var rps = match.RightPoints.ToArray().Take((int)(match.RightPoints.Size * takeBest)).ToArray();

            var F = ComputeMatrix.F(new VectorOfPointF(lps), new VectorOfPointF(rps));
            var K = EstimateCameraFromImagePair.K(F, left.Width, right.Height);
            var E = ComputeMatrix.E(F, K);
            FindTransformation.DecomposeToRT(E, out Image<Arthmetic, double> R, out Image<Arthmetic, double> t);
            PrintMatricesInfo(E, K, R, t);
        }

        private void DrawFeatures(Mat left, Mat right, MacthingResult match, double takeBest)
        {
            MatchDrawer.DrawFeatures(left, right, match, takeBest, macthedView);
            MatchDrawer.DrawCricles(leftView, left, match.LeftKps);
            MatchDrawer.DrawCricles(rightView, right, match.RightKps);
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
