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

using Matrix = Emgu.CV.Image<Egomotion.Arthmetic, double>;

namespace Egomotion
{
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
                if(!FindTwoViewsMatrices(lps_n, rps_n, K, out var F, out var E, out var R, out var t, out var X))
                {
                    return null;
                }

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
        
        public static bool FindTwoViewsMatrices(List<PointF> left, List<PointF> right, Matrix K, out Matrix F, out Matrix E, out Matrix R, out Matrix t, out Matrix X)
        {
            var lps_n = new List<PointF>(left);
            var rps_n = new List<PointF>(right);
            NormalizePoints2d(lps_n, out var NL);
            NormalizePoints2d(rps_n, out var NR);

            F = ComputeMatrix.F(new VectorOfPointF(lps_n.ToArray()), new VectorOfPointF(rps_n.ToArray()));
            if (F == null)
            {
                E = R = t = X = null;
                return false;
            }
            F = NR.T().Multiply(F).Multiply(NL);
            E = ComputeMatrix.E(F, K);

            DecomposeToRTAndTriangulate(left, right, K, E, out R, out t, out X);

            return true;
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
            Image<Arthmetic, double> R, Image<Arthmetic, double> t,
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

            var LP = Kinv.Multiply(Utils.Matrixify(left));
            var RP = Kinv.Multiply(Utils.Matrixify(right));

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
    }
 }
