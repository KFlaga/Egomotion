using Emgu.CV;
using Emgu.CV.Structure;
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
    public class FindTransformation
    {
        public static void FindFeatures(Mat image, Emgu.CV.Features2D.Feature2D detector, out MKeyPoint[] kps, out Mat descriptors)
        {
            kps = detector.Detect(image);

            Emgu.CV.Util.VectorOfKeyPoint vectorOfKp = new Emgu.CV.Util.VectorOfKeyPoint(kps);

            var desc = new Emgu.CV.XFeatures2D.BriefDescriptorExtractor(32);
            descriptors = new Mat();
            desc.Compute(image, vectorOfKp, descriptors);
        }

        public static Image<Arthmetic, double> ComputeFundametnalMatrix(Emgu.CV.Util.VectorOfPointF point1, Emgu.CV.Util.VectorOfPointF point2)
        {

            Mat F = Emgu.CV.CvInvoke.FindFundamentalMat(point1, point2);
            return F.ToImage<Arthmetic, double>();
        }

        public static void FindMatches(Emgu.CV.Util.VectorOfDMatch vector1, MKeyPoint[] kp1, MKeyPoint[] kp2, out Emgu.CV.Util.VectorOfPointF point1, out Emgu.CV.Util.VectorOfPointF point2 )
        {
            point1 = new Emgu.CV.Util.VectorOfPointF();
            point2 = new Emgu.CV.Util.VectorOfPointF();

                for (int j = 0; j<vector1.Size; j++)
                {
                point1.Push(new PointF[] { kp1[vector1[j].QueryIdx].Point });
                    point2.Push(new PointF[] { kp2[vector1[j].TrainIdx].Point });
                }
        }

        public static void ReturnRT(Image<Arthmetic, double> F, out Image<Arthmetic, double> R, out Image<Arthmetic, double> t)
        {
            Mat S = new Mat();
            Mat U = new Mat();
            Mat VT = new Mat();

            Emgu.CV.CvInvoke.SVDecomp(F, S, U, VT, Emgu.CV.CvEnum.SvdFlag.Default);

            var S2 = S.ToImage<Arthmetic, double>();
            var U2 = U.ToImage<Arthmetic, double>();
            var VT2 = VT.ToImage<Arthmetic, double>();

            Image<Arthmetic, double> W = new Image<Arthmetic, double>(new double[,,] {
                { {0}, {-1 }, {0 } } ,
                { {1}, {0 }, {0 } } ,
                { {0}, {0 }, {1 } } ,
            });

            var R1 = U2.Multiply(W.T()).Multiply(VT);
            var R2 = U2.Multiply(W).Multiply(VT);

            var I = new Image<Arthmetic, double>(3, 3);
            I[0, 0] = 1;
            I[1, 1] = 1;
            I[2, 2] = 1;

            var s1 = (R1 - I).Norm;
            var s2 = (R2 - I).Norm;

            R = s1 < s2 ? R1 : R2;


            double ss = (S2[0, 0] + S2[1, 0]) / 2;

            Image<Arthmetic, double> Z = new Image<Arthmetic, double>(new double[,,] {
                { {0}, {-ss }, {0 } } ,
                { {ss}, {0 }, {0 } } ,
                { {0}, {0 }, {0 } } ,
            });

            t = U2.Multiply(Z).Multiply(U.T());
           
        }

        public static OdometerFrame GetOdometerFrame(Mat left, Mat right, Emgu.CV.Features2D.Feature2D detector, Image<Arthmetic, double> K)
        {
            ReturnKF(left, right, detector, out Image<Arthmetic, double> F, out Emgu.CV.Util.VectorOfPointF point1, out Emgu.CV.Util.VectorOfPointF point2);

            if (F.Rows == 0)
            {
                return null;
            }
            //var K = Optimal(F.Mat, left.Width, left.Height);
            var E = K.T().Multiply(F).Multiply(K);
            ReturnRT(E, out Image<Arthmetic, double> R, out Image<Arthmetic, double> t);
            OdometerFrame odometerFrame = new OdometerFrame();
            odometerFrame.Rotation = RotationConverter.MatrixToEulerXYZ(R);
            odometerFrame.RotationMatrix = R;
            odometerFrame.Translation = new Image<Arthmetic, double>(1, 3);
            odometerFrame.Translation[0, 0] = t[2, 1];
            odometerFrame.Translation[1, 0] = t[0, 2];
            odometerFrame.Translation[2, 0] = t[1, 0];
         //   odometerFrame.Translation = R.T().Multiply(odometerFrame.Translation);
         //   odometerFrame.Translation = odometerFrame.Translation.Mul(1 / odometerFrame.Translation.Norm);
            odometerFrame.MatK = K;

            Image<Arthmetic, double> C = ComputeC(R, odometerFrame.Translation, K, point1[0], point2[0]);
            odometerFrame.Translation = R.Multiply(C);

            return odometerFrame;
        }

        public static void ReturnKF(Mat left, Mat right, Emgu.CV.Features2D.Feature2D detector, out Image<Arthmetic,double> F, out Emgu.CV.Util.VectorOfPointF point1, out Emgu.CV.Util.VectorOfPointF point2)
        {
            FindFeatures(left, detector, out MKeyPoint[] kps1, out Mat desc1);
            FindFeatures(right, detector, out MKeyPoint[] kps2, out Mat desc2);

            var matches = MakeMatches(desc1, desc2);

            FindMatches(matches, kps1, kps2, out point1, out point2);
            F = ComputeFundametnalMatrix(point1, point2);
            if (F.Rows == 0)
            {
                return;
            }

            F = F.Mul(1 / F.Norm);
           
        }

        public static Emgu.CV.Util.VectorOfDMatch MakeMatches(Mat desc1, Mat desc2)
        {
            Emgu.CV.Features2D.BFMatcher m = new Emgu.CV.Features2D.BFMatcher(Emgu.CV.Features2D.DistanceType.Hamming, true);
            Emgu.CV.Util.VectorOfDMatch matches = new Emgu.CV.Util.VectorOfDMatch();
            m.Match(desc1, desc2, matches);
            return matches;
        }

        public static Image<Arthmetic, double> MatrixE (Mat F, double fx, double fy, double px, double py)
        {
            Image<Arthmetic, double> K = new Image<Arthmetic, double>(new double[,,] {
                { {fx}, {0}, {px} } ,
                { {0}, {fy}, {py} } ,
                { {0}, {0 }, {1 } } ,
            });

            Image<Arthmetic, double> E = K.T().Multiply(F).Multiply(K);

            return E;
        }

        public static double CostK(Mat F, double fx, double fy, double px, double py, double w, double h)
        {
            Image<Arthmetic, double> E = MatrixE(F, fx, fy, px, py);

            Mat S = new Mat();
            Mat U = new Mat();
            Mat VT = new Mat();

            CvInvoke.SVDecomp(E, S, U, VT, Emgu.CV.CvEnum.SvdFlag.Default);

            var S2 = S.ToImage<Arthmetic, double>();

            double s1 = S2[0, 0];
            double s2 = S2[1, 0];
            double weight = 0.03;
            double errS = Math.Abs(Math.Abs(s1 / s2) - 1);
            double errF = Math.Abs(Math.Abs(fx / fy) - 1);
            double errPx = Math.Abs(Math.Abs(2 * px / w) - 1);
            double errPy = Math.Abs(Math.Abs(2 * py / h) - 1);

            return errS + weight * (errF + errPx + errPy);

        }

        public class ObjFunc : IObjectiveFunction
        {
            public Mat F;
            public double w;
            public double h;

            public Vector<double> Point { get; set; }

            public double Value { get; set; }

            public bool IsGradientSupported => true;

            public Vector<double> Gradient { get; set; }

            public bool IsHessianSupported => false;

            public MathNet.Numerics.LinearAlgebra.Matrix<double> Hessian => throw new NotImplementedException();

            public IObjectiveFunction CreateNew()
            {
                return new ObjFunc() { F = F, w = w, h = h };
            }

            public void EvaluateAt(Vector<double> point)
            {
                Point = point;
                Value = CostK(F, point[0], point[1], point[2], point[3], w, h);
                Gradient = grad(point);
            }

            public IObjectiveFunction Fork()
            {
                return new ObjFunc() { F = F, Point = Point, Value = Value, w = w, h = h };
            }

            private Vector<double> grad(Vector<double> point)
            {
                Vector<double> g = new DenseVector(point.Count);
                for (int i = 0; i < point.Count; ++i)
                {
                    double hh = 1e-3;
                    Vector<double> p1 = point.Clone();
                    p1[i] = point[i] - hh;
                    double v1 = CostK(F, p1[0], p1[1], p1[2], p1[3], w, h);
                    p1[i] = point[i] + hh;
                    double v2 = CostK(F, p1[0], p1[1], p1[2], p1[3], w, h);
                    g[i] = (v2 - v1) / (2 * hh);
                }
                return g;
            }
        }

        public static Image<Arthmetic, double> Optimal(Mat F, double width, double height)
        {
            double fi = (width + height) / 2;
            var minimizer = new BfgsMinimizer(1e-6, 1e-6, 1e-6);
            var result = minimizer.FindMinimum(new ObjFunc() { F = F, w = width, h = height },
              //  new DenseVector(new double[] { 0, 0, 0, 0 }),
             //   new DenseVector(new double[] { 1000, 1000, 1000, 1000 }),
                new DenseVector(new double[] { fi, fi, width / 2, height / 2 })
            );

            var p = result.MinimizingPoint;
            Image<Arthmetic, double> K = new Image<Arthmetic, double>(new double[,,] {
                { {p[0]}, {0}, {p[2]} } ,
                { {0}, {p[1]}, {p[3]} } ,
                { {0}, {0 }, {1 } } ,
            });
            return K;
        }

        public static Image<Arthmetic, double> smallOptimal(List<Image<Arthmetic, double>> F, double width, double height)
        {
            double fi = (width + height) / 2;
            var minimizer = new BfgsMinimizer(1e-6, 1e-6, 1e-6);
            var result = minimizer.FindMinimum(new ObjFunc2() { F = F, w = width, h = height },
                //  new DenseVector(new double[] { 0, 0, 0, 0 }),
                //   new DenseVector(new double[] { 1000, 1000, 1000, 1000 }),
                new DenseVector(new double[] { fi, fi, width / 2, height / 2 })
            );

            var p = result.MinimizingPoint;
            Image<Arthmetic, double> K = new Image<Arthmetic, double>(new double[,,] {
                { {p[0]}, {0}, {p[2]} } ,
                { {0}, {p[1]}, {p[3]} } ,
                { {0}, {0 }, {1 } } ,
            });
            return K;
        }

        public static double smallCostK(List<Image<Arthmetic, double>> F, double fx, double fy, double px, double py, double w, double h)
        {
            double errS=0;
            for (int s = 0; s<F.Count; s++)
            {

                Image<Arthmetic, double> E = MatrixE(F[s].Mat, fx, fy, px, py);
                Mat S = new Mat();
                Mat U = new Mat();
                Mat VT = new Mat();

                CvInvoke.SVDecomp(E, S, U, VT, Emgu.CV.CvEnum.SvdFlag.Default);

                var S2 = S.ToImage<Arthmetic, double>();

                double s1 = S2[0, 0];
                double s2 = S2[1, 0];

                if (s2 == 0)
                {
                    errS += 1;
                }
                else
                {
                    errS += (s1 - s2) / s2;
                }
             
      
            }

            return errS;

        }

        public class ObjFunc2 : IObjectiveFunction
        {
            public List<Image<Arthmetic, double>> F;
            public double w;
            public double h;

            public Vector<double> Point { get; set; }

            public double Value { get; set; }

            public bool IsGradientSupported => true;

            public Vector<double> Gradient { get; set; }

            public bool IsHessianSupported => false;

            public MathNet.Numerics.LinearAlgebra.Matrix<double> Hessian => throw new NotImplementedException();

            public IObjectiveFunction CreateNew()
            {
                return new ObjFunc2() { F = F, w = w, h = h };
            }

            public void EvaluateAt(Vector<double> point)
            {
                Point = point;
                Value = smallCostK(F, point[0], point[1], point[2], point[3], w, h);
                Gradient = grad(point);
            }

            public IObjectiveFunction Fork()
            {
                return new ObjFunc2() { F = F, Point = Point, Value = Value, w = w, h = h };
            }

            private Vector<double> grad(Vector<double> point)
            {
                Vector<double> g = new DenseVector(point.Count);
                for (int i = 0; i < point.Count; ++i)
                {
                    double hh = 1e-3;
                    Vector<double> p1 = point.Clone();
                    p1[i] = point[i] - hh;
                    double v1 = smallCostK(F, p1[0], p1[1], p1[2], p1[3], w, h);
                    p1[i] = point[i] + hh;
                    double v2 = smallCostK(F, p1[0], p1[1], p1[2], p1[3], w, h);
                    g[i] = (v2 - v1) / (2 * hh);
                }
                return g;
            }
        }


        public static Image<Arthmetic, double> ComputeK (List<Mat> mats, Emgu.CV.Features2D.Feature2D detector)
        {
            List<Image<Arthmetic, double>> F = new List<Image<Arthmetic, double>>();
            for ( int i = 0; i< mats.Count - 1; i += 2)
            {
                ReturnKF(mats[i], mats[i + 1], detector, out Image<Arthmetic, double> fs, out Emgu.CV.Util.VectorOfPointF point1, out Emgu.CV.Util.VectorOfPointF point2);
                F.Add(fs);
            }
            return smallOptimal(F, mats[0].Width, mats[0].Height);
        }

        public static Image<Arthmetic, double> ComputeC(Image<Arthmetic, double> R, Image<Arthmetic, double> T, Image<Arthmetic, double> K, PointF F1, PointF F2)
        {
            Image<Arthmetic, double> pointFr = new Image<Arthmetic, double>(new double[,,] {
                { {F1.X}} ,
                { {F1.Y}} ,
                { {1}} ,
            });
            Image<Arthmetic, double> pointSc = new Image<Arthmetic, double>(new double[,,] {
                { {F2.X}} ,
                { {F2.Y}} ,
                { {1}} ,
            });

            Image<Arthmetic, double> Kinv = new Image<Arthmetic, double>(3, 3);
            CvInvoke.Invert(K, Kinv, Emgu.CV.CvEnum.DecompMethod.LU);

            Image<Arthmetic, double> P1 = Kinv.Multiply(pointFr);
            Image<Arthmetic, double> P2 = Kinv.Multiply(pointSc);

            Image<Arthmetic, double> C = R.T().Multiply(T);

            double alfa = C[0, 0] / C[2, 0];
            double beta = C[1, 0] / C[2, 0];

            double cz = (R[0, 0] * P1[0, 0] + R[0, 1] * P1[1, 0] + R[0, 2] * P1[2, 0] - P2[0, 0]) / (R[0, 0] * alfa + R[0, 1] * beta + R[0, 2]);

            C[0, 0] = cz * alfa;
            C[1, 0] = cz * beta;
            C[2, 0] = cz;

            return C;
        }


    }
    }
