using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Optimization;
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

    public class FindTransformation
    {
        public static OdometerFrame GetOdometerFrame(Mat left, Mat right, Feature2D detector, Feature2D descriptor, DistanceType distanceType, double maxDistance, Image<Arthmetic, double> K, double takeBest = 1.0)
        {
            var match = MatchImagePair.Match(left, right, detector, descriptor, distanceType, maxDistance);

            var lps = match.LeftPointsList.Take((int)(match.LeftPoints.Size * takeBest));
            var rps = match.RightPointsList.Take((int)(match.RightPoints.Size * takeBest));

            var lps_n = lps.ToList();
            var rps_n = rps.ToList();
            var H = EstimateHomography(lps_n, rps_n, K);
            if(IsPureRotation(H, 0.02))
            {
                OdometerFrame odometerFrame = new OdometerFrame();
                odometerFrame.Rotation = RotationConverter.MatrixToEulerXYZ(H);
                odometerFrame.RotationMatrix = RotationConverter.EulerXYZToMatrix(odometerFrame.Rotation);
                odometerFrame.MatK = K;
                odometerFrame.Match = match;
                odometerFrame.Translation = new Image<Arthmetic, double>(1, 3);
                return odometerFrame;
            }
            else
            {
                FindTransformation.NormalizePoints2d(lps_n, out var NL);
                FindTransformation.NormalizePoints2d(rps_n, out var NR);

                var F = ComputeMatrix.F(new VectorOfPointF(lps_n.ToArray()), new VectorOfPointF(rps_n.ToArray()));
                if (F == null)
                {
                    return null;
                }
                F = NR.T().Multiply(F).Multiply(NL);

                var E = ComputeMatrix.E(F, K);
                DecomposeToRTAndTriangulate(lps.ToList(), rps.ToList(), K, E,
                    out Image<Arthmetic, double> R, out Image<Arthmetic, double> t, out Image<Arthmetic, double> X);

                OdometerFrame odometerFrame = new OdometerFrame();
                odometerFrame.Rotation = RotationConverter.MatrixToEulerXYZ(R);
                odometerFrame.RotationMatrix = R;
                odometerFrame.MatK = K;
                odometerFrame.Match = match;

                Image<Arthmetic, double> C = R.T().Multiply(t).Mul(-1);
                odometerFrame.Translation = C.Mul(1.0 / C.Norm);
                return odometerFrame;
            }
        }

        public static void DecomposeToRT(Image<Arthmetic, double> E,
            out Image<Arthmetic, double>[] Rs,
            out Image<Arthmetic, double>[] ts)
        {
            var svd = new Svd(E);

            Image<Arthmetic, double> W = new Image<Arthmetic, double>(new double[,,] {
                { {0}, {-1 }, {0 } } ,
                { {1}, {0 }, {0 } } ,
                { {0}, {0 }, {1 } } ,
            });

            var R1 = svd.U.Multiply(W.T()).Multiply(svd.VT);
            double det1 = CvInvoke.Determinant(R1);
            if (det1 < 0)
                R1 = R1.Mul(-1);

            var R2 = svd.U.Multiply(W).Multiply(svd.VT);
            double det2 = CvInvoke.Determinant(R2);
            if (det2 < 0)
                R2 = R2.Mul(-1);
            
            double ss = (svd.S[0, 0] + svd.S[1, 0]) / 2;

            Image<Arthmetic, double> Z = new Image<Arthmetic, double>(new double[,,] {
                { {0}, {-ss }, {0 } } ,
                { {ss}, {0 }, {0 } } ,
                { {0}, {0 }, {0 } } ,
            });

            var t1 = svd.U.Multiply(Z).Multiply(svd.U.T());
            t1 = ComputeMatrix.CrossProductToVector(t1);
            var t2 = t1.Mul(-1);

            Rs = new Image<Arthmetic, double>[] { R1, R2 };
            ts = new Image<Arthmetic, double>[] { t1, t2 };
        }

        public static void DecomposeToRTAndTriangulate(
            List<PointF> left, List<PointF> right, Image<Arthmetic, double> K, Image<Arthmetic, double> E,
            out Image<Arthmetic, double> R, out Image<Arthmetic, double> t, out Image<Arthmetic, double> pts3d)
        {
            pts3d = null;
            R = null;
            t = null;

            DecomposeToRT(E, out Image<Arthmetic, double>[] Rs, out Image<Arthmetic, double>[] ts);

            int maxInliners = -1;
            for(int i = 0; i < Rs.Length; ++i)
            {
                for(int j = 0; j < ts.Length; ++j)
                {
                    int inliners = TriangulateChieral(left, right, K, Rs[i], ts[j], out Image<Arthmetic, double> X);
                    if(inliners > maxInliners)
                    {
                        maxInliners = inliners;
                        pts3d = X;
                        R = Rs[i];
                        t = ts[i];
                    }
                }
            }
        }

        public static int TriangulateChieral(
            List<PointF> left, List<PointF> right, Image<Arthmetic, double> K,
            Image<Arthmetic, double> R,Image<Arthmetic, double> t,
            out Image<Arthmetic, double> pts3d)
        {
            // init 3d point matrix
            pts3d = new Image<Arthmetic, double>(left.Count, 4);

            // init projection matrices
            var P1 = ComputeMatrix.Camera(K);
            var P2 = ComputeMatrix.Camera(K, R, t);

            // Transform points lists into matrices
            var img1 = new Image<Arthmetic, double>(left.Count, 2);
            var img2 = new Image<Arthmetic, double>(left.Count, 2);
            for (int i = 0; i < left.Count; ++i)
            {
                img1[0, i] = left[i].X;
                img1[1, i] = left[i].Y;
                img2[0, i] = right[i].X;
                img2[1, i] = right[i].Y;
            }

            CvInvoke.TriangulatePoints(P1, P2, img1, img2, pts3d);

            // Scale points, so that W = 1
            for (int i = 0; i < left.Count; ++i)
            {
                pts3d[0, i] = pts3d[0, i] / pts3d[3, i];
                pts3d[1, i] = pts3d[1, i] / pts3d[3, i];
                pts3d[2, i] = pts3d[2, i] / pts3d[3, i];
                pts3d[3, i] = pts3d[3, i] / pts3d[3, i];
            }

            // compute points in front of camera (TODO: how does it work?)
            var AX1 = P1.Multiply(pts3d);
            var BX1 = P2.Multiply(pts3d);
            int num = 0;
            for (int i = 0; i < left.Count; i++)
                if (AX1[2, i] * pts3d[3, i] > 0 && BX1[2, i] * pts3d[3, i] > 0)
                    num++;
            
            return num;
        }

        public static void NormalizePoints2d(List<PointF> pts, out Image<Arthmetic, double> N)
        {
            // Compute centroid of both point sets
            float mean_x = 0, mean_y = 0;
            for (int i = 0; i < pts.Count; ++i)
            {
                mean_x += pts[i].X;
                mean_y += pts[i].Y;
            }
            mean_x /= pts.Count;
            mean_y /= pts.Count;

            // Shift origins to centroids
            for (int i = 0; i < pts.Count; ++i)
            {
                pts[i] = PointF.Subtract(pts[i], new SizeF(mean_x, mean_y));
            }

            // Scale points so that mean distance from origin is sqrt(2)
            float scale = 0;
            for (int i = 0; i < pts.Count; ++i)
            {
                scale += (float)Math.Sqrt(pts[i].X * pts[i].X + pts[i].Y * pts[i].Y);
            }

            float targetMean = (float)Math.Sqrt(2.0);
            scale = targetMean * pts.Count / scale;
            for (int i = 0; i < pts.Count; ++i)
            {
                pts[i] = new PointF(pts[i].X * scale, pts[i].Y * scale);
            }

            // compute corresponding transformation matrices
            N = new Image<Arthmetic, double>(new double[,,] {
                { {scale}, {0}, {-scale * mean_x}, } ,
                { {0}, {scale}, {-scale * mean_y}, } ,
                { {0}, {0}, {1}, } ,
            });
        }

        public static Image<Arthmetic, double> EstimateHomography(List<PointF> left, List<PointF> right, Image<Arthmetic, double> K)
        {
            var Kinv = new Image<Arthmetic, double>(3, 3);
            CvInvoke.Invert(K, Kinv, Emgu.CV.CvEnum.DecompMethod.Svd);

            var LP = Kinv.Multiply(Errors.Matrixify(left));
            var RP = Kinv.Multiply(Errors.Matrixify(right));

            PointF[] lps = new PointF[left.Count];
            PointF[] rps = new PointF[left.Count];
            for (int i = 0; i < left.Count; ++i)
            {
                lps[i].X = (float)(LP[0, i] / LP[2, i]);
                lps[i].Y = (float)(LP[1, i] / LP[2, i]);
                rps[i].X = (float)(RP[0, i] / RP[2, i]);
                rps[i].Y = (float)(RP[1, i] / RP[2, i]);
            }

            var h = CvInvoke.FindHomography(lps, rps, Emgu.CV.CvEnum.RobustEstimationAlgorithm.LMEDS);
            return h.ToImage<Arthmetic, double>();
        }

        public static bool IsPureRotation(Image<Arthmetic, double> H, double threshold = 0.05)
        {
            var svd = new Svd(H);
            // If this is rotation all eigenvalues should be close to 1.0
            double ratio = svd.S[2, 0] / svd.S[0, 0];
            if(ratio < 1.0 - threshold)
            {
                return false;
            }
            if(Math.Abs(svd.S[0, 0] - 1.0) > threshold)
            {
                return false;
            }
            return true;
        }

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

            var match12 = MatchImagePair.Match(left, right, detector, descriptor, distanceType, maxDistance);
            var match23 = MatchImagePair.Match(left, right, detector, descriptor, distanceType, maxDistance);
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

            if(Cs[0].Norm < 1e-8 || Cs[1].Norm < 1e-8)
            {
                // TODO: alternative for such case
                //throw new NotImplementedException("Initial camera center has zero elements");
                return null;
            }

            double scale = 0.0;
            for (int i = 0; i < match.Left.Count; ++i)
            {
                List<double> Ls = new List<double>();

                for(int c = 0; c < 2; ++c)
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
