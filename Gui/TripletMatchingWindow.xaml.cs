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
    public partial class TripletMatchingWindow : Window
    {
        public TripletMatchingWindow()
        {
            InitializeComponent();
        }

        double pointErr = 25.1;

        private bool AreEqual(PointF p1, PointF p2)
        {
            return Math.Abs(p1.X - p2.X) < pointErr && Math.Abs(p1.Y - p2.Y) < pointErr;
        }

        private int IndexOf_X(List<MatchClosePoints.Item> ps, PointF p)
        {
            int closest = MatchClosePoints.FindBestMatch(new MatchClosePoints.Item() { pos = p }, ps, (a, b) =>
            {
                double dx = a.pos.X - b.pos.X;
                double dy = a.pos.Y - b.pos.Y;
                return dx * dx + dy * dy;
            }, 20.0);
            if (closest == -1)
                return -1;
            return ps[closest].index;
        }
        
        public void ProcessImages(Mat left, Mat middle, Mat right, Feature2D detector, Feature2D descriptor, DistanceType distance)
        {
            double maxDistance = 20.0;
            var match12 = MatchImagePair.Match(left, middle, detector, descriptor, distance, maxDistance);
            var match23 = MatchImagePair.Match(middle, right, detector, descriptor, distance, maxDistance);
            var match13 = MatchImagePair.Match(left, right, detector, descriptor, distance, maxDistance);

            TripletMatch tmatch = new TripletMatch();

            List<MDMatch> m12 = new List<MDMatch>();
            List<MDMatch> m23 = new List<MDMatch>();

            var left1 = match12.LeftPoints;
            var right1 = match12.RightPoints;
            var left2 = match23.LeftPoints;
            var left2_X = MatchClosePoints.SortByX(match23.LeftPoints);
            var right2 = match23.RightPoints;
            var left3 = match13.LeftPoints;
            var right3 = match13.RightPoints;
            var right3_X = MatchClosePoints.SortByX(match13.LeftPoints);

            for (int idx12 = 0; idx12 < left1.Size; ++idx12)
            {
                var p1 = left1[idx12];
                var p2 = right1[idx12];
                int idx23 = IndexOf_X(left2_X, p2);
                if(idx23 != -1)
                {
                    var p3 = right2[idx23];
                    int idx13 = IndexOf_X(right3_X, p1);
                    if(idx13 != -1)
                    {
                        if(AreEqual(left1[idx12], left3[idx13]))
                        {
                            tmatch.Left.Add(p1);
                            tmatch.Middle.Add(p2);
                            tmatch.Right.Add(p3);

                            m12.Add(match12.Matches[idx12]);
                            m23.Add(match23.Matches[idx23]);
                        }
                    }
                }
            }

            match12.Matches = new VectorOfDMatch(m12.ToArray());
            match23.Matches = new VectorOfDMatch(m23.ToArray());

            MatchDrawer.DrawFeatures(left, right, match12, 1.0, bottomView);
            MatchDrawer.DrawFeatures(left, right, match23, 1.0, upperView);

            var F12 = ComputeMatrix.F(new VectorOfPointF(tmatch.Left.ToArray()), new VectorOfPointF(tmatch.Middle.ToArray()));
            var F23 = ComputeMatrix.F(new VectorOfPointF(tmatch.Middle.ToArray()), new VectorOfPointF(tmatch.Right.ToArray()));
            var F13 = ComputeMatrix.F(new VectorOfPointF(tmatch.Left.ToArray()), new VectorOfPointF(tmatch.Right.ToArray()));

            if(F12 == null || F23 == null || F13 == null)
            {
                info.Text = "Too few matches";
                return;
            }

            var Fs = new List<Image<Arthmetic, double>> { F12, F23, F13 };

            var K = EstimateCameraFromImageSequence.K(Fs, left.Width, right.Height);

            var Es = new List<Image<Arthmetic, double>>
            {
                ComputeMatrix.E(F12, K),
                ComputeMatrix.E(F23, K),
                ComputeMatrix.E(F13, K)
            };

            FindTransformation.DecomposeToRT(Es[0], out Image<Arthmetic, double> R12, out Image<Arthmetic, double> t12);
            FindTransformation.DecomposeToRT(Es[1], out Image<Arthmetic, double> R23, out Image<Arthmetic, double> t23);
            FindTransformation.DecomposeToRT(Es[2], out Image<Arthmetic, double> R13, out Image<Arthmetic, double> t13);

            var Rs = new List<Image<Arthmetic, double>>
            {
                RotationConverter.MatrixToEulerXYZ(R12),
                RotationConverter.MatrixToEulerXYZ(R23),
                RotationConverter.MatrixToEulerXYZ(R13)
            };
            var ts = new List<Image<Arthmetic, double>>
            {
                ComputeMatrix.CrossProductToVector(t12),
                ComputeMatrix.CrossProductToVector(t23),
                ComputeMatrix.CrossProductToVector(t13)
            };

            PrintMatricesInfo(Es, K, Rs, ts);
        }

        private void PrintMatricesInfo(List<Image<Arthmetic, double>> Es, Image<Arthmetic, double> K, List<Image<Arthmetic, double>> Rs, List<Image<Arthmetic, double>> ts)
        {
            StringBuilder sb = new StringBuilder();

            Svd svd12 = new Svd(Es[0]);
            Svd svd23 = new Svd(Es[1]);
            Svd svd13 = new Svd(Es[2]);
            sb.AppendLine(string.Format("E12 s1 = {0}, s2 = {1}", svd12.S[0, 0], svd12.S[1, 0]));
            sb.AppendLine(string.Format("E23 s1 = {0}, s2 = {1}", svd23.S[0, 0], svd23.S[1, 0]));
            sb.AppendLine(string.Format("E13 s1 = {0}, s2 = {1}", svd13.S[0, 0], svd13.S[1, 0]));

            sb.AppendLine();
            sb.AppendLine(string.Format("fx = {0}, fy = {1}", K[0, 0], K[1, 1]));
            sb.AppendLine(string.Format("px = {0}, py = {1}", K[0, 2], K[1, 2]));
            
            sb.AppendLine();
            sb.AppendLine(string.Format("t12 = {0}, {1}, {2}", ts[0][0, 0], ts[0][1, 0], ts[0][2, 0]));
            sb.AppendLine(string.Format("t23 = {0}, {1}, {2}", ts[1][0, 0], ts[1][1, 0], ts[1][2, 0]));
            sb.AppendLine(string.Format("t13 = {0}, {1}, {2}", ts[2][0, 0], ts[2][1, 0], ts[2][2, 0]));
            
            sb.AppendLine();
            sb.AppendLine(string.Format("r12 = {0}, {1}, {2}", Rs[0][0, 0], Rs[0][1, 0], Rs[0][2, 0]));
            sb.AppendLine(string.Format("r23 = {0}, {1}, {2}", Rs[1][0, 0], Rs[1][1, 0], Rs[1][2, 0]));
            sb.AppendLine(string.Format("r13 = {0}, {1}, {2}", Rs[2][0, 0], Rs[2][1, 0], Rs[2][2, 0]));

            info.Text = sb.ToString();
        }
    }
}
