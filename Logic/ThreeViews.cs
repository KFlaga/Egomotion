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

namespace Egomotion
{
    public class TripletMatch
    {
        public List<PointF> Left = new List<PointF>();
        public List<PointF> Middle = new List<PointF>();
        public List<PointF> Right = new List<PointF>();

        public VectorOfDMatch Match12 = new VectorOfDMatch();
        public VectorOfDMatch Match23 = new VectorOfDMatch();
    }

    public static class ThreeViews
    {
        private static bool AreEqual(PointF p1, PointF p2, double pointErr)
        {
            return Math.Abs(p1.X - p2.X) < pointErr && Math.Abs(p1.Y - p2.Y) < pointErr;
        }

        private static int IndexOf_X(List<MatchClosePoints.Item> ps, PointF p)
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

        public static OdometerFrame GetOdometerFrame3(
            Mat left, Mat middle, Mat right, double lastScale, out double thisScale,
            Feature2D detector, Feature2D descriptor, DistanceType distanceType, double maxDistance,
            Image<Arthmetic, double> K, double takeBest = 1.0)
        {
            thisScale = lastScale;

            var match12 = MatchImagePair.Match(left, middle, detector, descriptor, distanceType, maxDistance);
            var match23 = MatchImagePair.Match(middle, right, detector, descriptor, distanceType, maxDistance);
            var match13 = MatchImagePair.Match(left, right, detector, descriptor, distanceType, maxDistance);

            var left1 = match12.LeftPoints;
            var right1 = match12.RightPoints;
            var left2 = match23.LeftPoints;
            var left2_X = MatchClosePoints.SortByX(match23.LeftPoints);
            var right2 = match23.RightPoints;
            var left3 = match13.LeftPoints;
            var right3 = match13.RightPoints;
            var right3_X = MatchClosePoints.SortByX(match13.LeftPoints);

            TripletMatch tmatch = new TripletMatch();

            List<MDMatch> m12 = new List<MDMatch>();
            List<MDMatch> m23 = new List<MDMatch>();

            for (int idx12 = 0; idx12 < left1.Size; ++idx12)
            {
                var p1 = left1[idx12];
                var p2 = right1[idx12];
                int idx23 = IndexOf_X(left2_X, p2);
                if (idx23 != -1)
                {
                    var p3 = right2[idx23];
                    int idx13 = IndexOf_X(right3_X, p1);
                    if (idx13 != -1)
                    {
                        if (AreEqual(left1[idx12], left3[idx13], maxDistance))
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

            var F12 = ComputeMatrix.F(new VectorOfPointF(tmatch.Left.ToArray()), new VectorOfPointF(tmatch.Middle.ToArray()));
            //  var F23 = ComputeMatrix.F(new VectorOfPointF(tmatch.Middle.ToArray()), new VectorOfPointF(tmatch.Right.ToArray()));
            var F13 = ComputeMatrix.F(new VectorOfPointF(tmatch.Left.ToArray()), new VectorOfPointF(tmatch.Right.ToArray()));

            if (F12 == null || F13 == null)
            {
                return null;
            }

            var Es = new List<Image<Arthmetic, double>>
            {
                ComputeMatrix.E(F12, K),
              //  ComputeMatrix.E(F23, K),
                ComputeMatrix.E(F13, K)
            };

            FindTransformation.DecomposeToRTAndTriangulate(tmatch.Left, tmatch.Middle, K, Es[0],
                out Image<Arthmetic, double> R12, out Image<Arthmetic, double> t12, out Image<Arthmetic, double> X12);
            // FindTransformation.DecomposeToRT(Es[1], out Image<Arthmetic, double> R23, out Image<Arthmetic, double> t23);
            FindTransformation.DecomposeToRTAndTriangulate(tmatch.Left, tmatch.Right, K, Es[1],
                out Image<Arthmetic, double> R13, out Image<Arthmetic, double> t13, out Image<Arthmetic, double> X13);

            var Rs = new List<Image<Arthmetic, double>>
            {
                R12,
                R13
            };
            var ts = new List<Image<Arthmetic, double>>
            {
                t12,
                t13
            };

            var cc = ComputeCameraCenter3(K, Rs, ts, tmatch);

            OdometerFrame odometerFrame = new OdometerFrame();
            odometerFrame.Rotation = RotationConverter.MatrixToEulerXYZ(Rs[0]);
            odometerFrame.RotationMatrix = Rs[0];
            odometerFrame.MatK = K;
            odometerFrame.Match = match12;

            //    Image<Arthmetic, double> C = ComputeCameraCenter(R, t, K, match);
            //  odometerFrame.Translation = R.Multiply(C);
            //   odometerFrame.Translation = R.T().Multiply(odometerFrame.Translation);
            odometerFrame.Translation = ts[0].Mul(lastScale / ts[0].Norm);
            odometerFrame.Center = lastScale * cc.C12;
            thisScale = cc.Ratio3To2;

            return odometerFrame;
        }

        public class ScaledCenter
        {
            public Image<Arthmetic, double> C12;
            public Image<Arthmetic, double> C13;
            public double Ratio3To2;
        }

        public static ScaledCenter ComputeCameraCenter3(
            Image<Arthmetic, double> K,
            List<Image<Arthmetic, double>> Rs, // R12, R13
            List<Image<Arthmetic, double>> ts, // t12, t13
            TripletMatch match)
        {
            Image<Arthmetic, double> Kinv = new Image<Arthmetic, double>(3, 3);
            CvInvoke.Invert(K, Kinv, Emgu.CV.CvEnum.DecompMethod.LU);

            var Cs = new List<Image<Arthmetic, double>>()
            {
                Rs[0].T().Multiply(ts[0]),
                Rs[1].T().Multiply(ts[1]),
            };

            if (Cs[0].Norm < 1e-8 || Cs[1].Norm < 1e-8)
            {
                // TODO: alternative for such case
                //throw new NotImplementedException("Initial camera center has zero elements");
                return null;
            }

            double scale = 0.0;
            for (int i = 0; i < match.Left.Count; ++i)
            {
                List<double> Ls = new List<double>();

                for (int c = 0; c < 2; ++c)
                {
                    var C = Cs[c];

                    var p1 = match.Left[i];
                    var p2 = c == 0 ? match.Middle[i] : match.Right[i];

                    double L = ComputeCameraCenterRatioForPoint(K, Kinv, Rs[c], C, p1, p2);
                    Ls.Add(L);
                }

                double scale_ = Ls[1] / Ls[0];
                var C12 = Cs[0].Mul(1.0 / Cs[0].Norm);
                var C13 = Cs[1].Mul(scale_ / Cs[1].Norm);
                var C23est = C13.Sub(C12);
                var scaleOrg = ts[1].Norm / ts[0].Norm;
                scale += scale_;
            }
            scale /= match.Left.Count;

            var C12_ = Cs[0].Mul(1 / Cs[0].Norm);
            var C13_ = Cs[1].Mul(scale / Cs[1].Norm);
            var C23est_ = C13_.Sub(C12_);

            return new ScaledCenter()
            {
                C12 = Cs[0].Mul(1 / Cs[0].Norm),
                C13 = Cs[1].Mul(scale / Cs[1].Norm),
                Ratio3To2 = scale
            };
        }

        public static double ComputeCameraCenterRatioForPoint(
            Image<Arthmetic, double> K,
            Image<Arthmetic, double> Kinv,
            Image<Arthmetic, double> R,
            Image<Arthmetic, double> C,
            PointF p1, PointF p2)
        {
            // Cast p1 and p2 into normalized reference frame
            Image<Arthmetic, double> P1 = Kinv.Multiply(p1.ToVector());
            Image<Arthmetic, double> P2 = R.T().Multiply(Kinv.Multiply(p2.ToVector()));
            double x1 = P1[0, 0] / P1[2, 0];
            double y1 = P1[1, 0] / P1[2, 0];
            double x2 = P2[0, 0] / P2[2, 0];
            double y2 = P2[1, 0] / P2[2, 0];

            // Scale C so that ||Cn|| = 1
            Image<Arthmetic, double> Cn = C.Mul(1 / C.Norm);

            // ||C|| / Z = (x2 - x1) / (x2*Cn_z - Cn_x) = (y2 - y1) / (y2*Cn_z - Cn_y) 
            double Lx = x2 - x1;
            double Mx = x2 * Cn[2, 0] - Cn[0, 0];

            double Ly = y2 - y1;
            double My = y2 * Cn[2, 0] - Cn[1, 0];

            double L1 = Lx / Mx;
            double L2 = Ly / My;

            return (L1 + L2) / 2; // cz = Z * L
        }
    }
}
