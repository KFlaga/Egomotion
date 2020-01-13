using Emgu.CV;
using MathNet.Numerics.Optimization;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Egomotion
{
    public class Errors
    {
        public static void ReprojectionError(
            Image<Arthmetic, double> estReal, List<PointF> img,
            Image<Arthmetic, double> K, Image<Arthmetic, double> R, Image<Arthmetic, double> t,
            out double mean, out double median, out List<double> errors)
        {
            var P = ComputeMatrix.Camera(K, R, t);

            var estImg = P.Multiply(estReal);

            errors = new List<double>();
            for (int i = 0; i < img.Count; ++i)
            {
                var estPoint = new Image<Arthmetic, double>(new double[,,]
                {
                    {{estImg[0, i] / estImg[2, i]}}, {{estImg[1, i] / estImg[2, i]}},
                });
                var realPoint = new Image<Arthmetic, double>(new double[,,]
                {
                    {{img[i].X}}, {{img[i].Y}},
                });

                errors.Add(estPoint.Sub(realPoint).Norm);
            }
            mean = errors.Sum() / errors.Count;
            median = errors[errors.Count / 2];
        }

        public static void TraingulationError(
            List<Image<Arthmetic, double>> ptsReal, Image<Arthmetic, double> estReal,
            out double mean, out double median, out List<double> errors)
        {
            errors = new List<double>();
            for (int i = 0; i < ptsReal.Count; ++i)
            {
                var estPoint = new Image<Arthmetic, double>(new double[,,]
                {
                    {{estReal[0, i]}}, {{estReal[1, i]}}, {{estReal[2, i]}},
                });
                var realPoint = new Image<Arthmetic, double>(new double[,,]
                {
                    {{ptsReal[i][0, 0]}}, {{ptsReal[i][1, 0]}}, {{ptsReal[i][2, 0]}},
                });

                var p1 = estPoint.Mul(1 / estPoint.Norm);
                var p2 = realPoint.Mul(1 / realPoint.Norm);

                errors.Add(p1.Sub(p2).Norm);
            }
            mean = errors.Sum() / errors.Count;
            median = errors[errors.Count / 2];
        }

        public static void ReprojectionError2d(
            List<PointF> left, List<PointF> right,
            Image<Arthmetic, double> K, Image<Arthmetic, double> R,
            out double mean, out double median, out List<double> errors)
        {
            var Kinv = new Image<Arthmetic, double>(3, 3);
            CvInvoke.Invert(K, Kinv, Emgu.CV.CvEnum.DecompMethod.Svd);

            var LP = Kinv.Multiply(Matrixify(left));
            var RP = Kinv.Multiply(Matrixify(right));

            var estRP = R.Multiply(LP);

            errors = new List<double>();
            for (int i = 0; i < left.Count; ++i)
            {
                var estPoint = new Image<Arthmetic, double>(new double[,,]
                {
                    {{estRP[0, i] / estRP[2, i]}}, {{estRP[1, i] / estRP[2, i]}},
                });
                var realPoint = new Image<Arthmetic, double>(new double[,,]
                {
                    {{RP[0, i] / RP[2, i]}}, {{RP[1, i] / RP[2, i]}},
                });

                errors.Add(estPoint.Sub(realPoint).Norm);
            }
            mean = errors.Sum() / errors.Count;
            median = errors[errors.Count / 2];
        }

        public static void ReprojectionError2dWithT(
            List<PointF> left, List<PointF> right,
            Image<Arthmetic, double> K, Image<Arthmetic, double> R, Image<Arthmetic, double> t,
            out double scale,
            out double mean, out double median, out List<double> errors)
        {
            var Kinv = new Image<Arthmetic, double>(3, 3);
            CvInvoke.Invert(K, Kinv, Emgu.CV.CvEnum.DecompMethod.Svd);

            var LP = Kinv.Multiply(Matrixify(left));
            var RP = Kinv.Multiply(Matrixify(right));
            
            Rep2dTError errorFunc = new Rep2dTError()
            {
                LP = LP,
                RP = RP,
                t = t,
                R = R
            };

            if(t.Norm < 1e-8)
            {
                scale = 0.0;
            }
            else
            {
                GoldenSectionMinimizer minimizer = new GoldenSectionMinimizer(1e-3);
                try
                {
                    var result = minimizer.FindMinimum(errorFunc, 0.0, 1e9);
                    scale = result.MinimizingPoint;
                }
                catch (Exception)
                {
                    scale = 0.0;
                }
            }
            
            errors = new List<double>();
            for (int i = 0; i < left.Count; ++i)
            {
                errors.Add(errorFunc.Error(scale, i));
            }
            mean = errors.Sum() / errors.Count;
            median = errors[errors.Count / 2];

            if(scale != 0.0)
            {
                scale = 1 / scale;
            }
        }

        public static Image<Arthmetic, double> Matrixify(List<Image<Arthmetic, double>> pts)
        {
            Image<Arthmetic, double> X = new Image<Arthmetic, double>(pts.Count, pts[0].Rows);
            for (int i = 0; i < pts.Count; ++i)
            {
                for (int j = 0; j < pts[0].Rows; ++j)
                {
                    X[j, i] = pts[i][j, 0];
                }
            }
            return X;
        }

        public static Image<Arthmetic, double> Matrixify(List<PointF> pts)
        {
            Image<Arthmetic, double> X = new Image<Arthmetic, double>(pts.Count, 3);
            for (int i = 0; i < pts.Count; ++i)
            {
                X[0, i] = pts[i].X;
                X[1, i] = pts[i].Y;
                X[2, i] = 1;
            }
            return X;
        }

        private class Rep2dTError : IScalarObjectiveFunction
        {
            public bool IsDerivativeSupported => false;
            public bool IsSecondDerivativeSupported => false;

            public Image<Arthmetic, double> LP { get; set; }
            public Image<Arthmetic, double> RP { get; set; }

            public Image<Arthmetic, double> t { get; set; }
            public Image<Arthmetic, double> R { get; set; }

            public class Evaluator : IScalarObjectiveFunctionEvaluation
            {
                public double Point { get; set; }
                public double Value { get; set; }
                public double Derivative { get; set; }
                public double SecondDerivative { get; set; }
            }

            public IScalarObjectiveFunctionEvaluation Evaluate(double point)
            {
                return new Evaluator()
                {
                    Point = point,
                    Value = Error(point)
                };
            }

            public double Error(double scale)
            {
                if (scale < 0.0)
                    return LP.Cols * 10000.0;

                double e = 0.0;
                for(int i = 0; i < LP.Cols; ++i)
                {
                    e += Error(scale, i);
                }
                return e;
            }

            public double Error(double scale, int index)
            {
                double x1 = LP[0, index] / LP[2, index];
                double y1 = LP[1, index] / LP[2, index];
                double x2 = RP[0, index] / RP[2, index];
                double y2 = RP[1, index] / RP[2, index];

                double r1 = R[0, 0] * x1 + R[0, 1] * y1 + R[0, 2];
                double r2 = R[1, 0] * x1 + R[1, 1] * y1 + R[1, 2];
                double r3 = R[2, 0] * x1 + R[2, 1] * y1 + R[2, 2];

                double Lx = r1 + scale * t[0, 0];
                double Ly = r2 + scale * t[1, 0];
                double Lz = r3 + scale * t[2, 0];

                double x2e = Lx / Lz;
                double y2e = Ly / Lz;

                return Math.Sqrt((x2 - x2e) * (x2 - x2e) + (y2 - y2e) * (y2 - y2e));
            }
        }
    }
}
