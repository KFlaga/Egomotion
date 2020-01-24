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
            Image<Arthmetic, double> ptsReal, Image<Arthmetic, double> estReal,
            out double mean, out double median, out List<double> errors)
        {
            errors = new List<double>();
            for (int i = 0; i < ptsReal.Cols; ++i)
            {
                var estPoint = new Image<Arthmetic, double>(new double[,,]
                {
                    {{estReal[0, i] / estReal[3, i]}}, {{estReal[1, i] / estReal[3, i]}}, {{estReal[2, i] / estReal[3, i]}},
                });
                var realPoint = new Image<Arthmetic, double>(new double[,,]
                {
                    {{ptsReal[0, i] / ptsReal[3, i]}}, {{ptsReal[1, i] / ptsReal[3, i]}}, {{ptsReal[2, i] / ptsReal[3, i]}},
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

            var LP = Kinv.Multiply(Utils.Matrixify(left));
            var RP = Kinv.Multiply(Utils.Matrixify(right));

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
    }

    public static class Utils
    {
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

        public static Image<Arthmetic, double> Matrixify(IEnumerable<PointF> pts)
        {
            Image<Arthmetic, double> X = new Image<Arthmetic, double>(pts.Count(), 3);
            int i = 0;
            foreach(PointF p in pts)
            {
                X[0, i] = p.X;
                X[1, i] = p.Y;
                X[2, i] = 1;
                ++i;
            }
            return X;
        }

        public static Image<Arthmetic, double> PutRTo4x4(Image<Arthmetic, double> R)
        {
            Image<Arthmetic, double> X = new Image<Arthmetic, double>(4, 4);
            for(int i = 0; i < 3; ++i)
            {
                for(int j = 0; j < 3; ++j)
                {
                    X[i, j] = R[i, j];
                }
            }
            X[3, 3] = 1.0;
            return X;
        }

        public static List<Image<Arthmetic, double>> Listify(Image<Arthmetic, double> pts, bool skipLast)
        {
            List<Image<Arthmetic, double>> X = new List<Image<Arthmetic, double>>(pts.Cols);
            for (int i = 0; i < pts.Cols; ++i)
            {
                if(skipLast)
                {
                    Image<Arthmetic, double> x = new Image<Arthmetic, double>(new double[,,]
                    {
                        {{ pts[0, i] / pts[3, i] }}, {{ pts[1, i] / pts[3, i] }}, {{ pts[2, i] / pts[3, i] }},
                    });
                    X.Add(x);
                }
                else
                {
                    Image<Arthmetic, double> x = new Image<Arthmetic, double>(new double[,,]
                    {
                        {{ pts[0, i] }}, {{ pts[1, i] }}, {{ pts[2, i] }}, {{ pts[3, i] }}
                    });
                    X.Add(x);
                }
            }
            return X;
        }
        
        public static Image<Arthmetic, double> Rx(double a)
        {
            double rad = a * Math.PI / 180.0;
            double ca = Math.Cos(rad);
            double sa = Math.Sin(rad);

            return new Image<Arthmetic, double>(new double[,,] {
                { {1}, {0}, {0} } ,
                { {0}, {ca}, {-sa} } ,
                { {0}, {sa}, {ca} } ,
            });
        }

        public static Image<Arthmetic, double> Rz(double a)
        {
            double rad = a * Math.PI / 180.0;
            double ca = Math.Cos(rad);
            double sa = Math.Sin(rad);

            return new Image<Arthmetic, double>(new double[,,] {
                { {ca}, {-sa}, {0} } ,
                { {sa}, {ca}, {0} } ,
                { {0}, {0}, {1} } ,
            });
        }

        public static Image<Arthmetic, double> I()
        {
            return new Image<Arthmetic, double>(new double[,,] {
                { {1}, {0}, {0} } ,
                { {0}, {1}, {0} } ,
                { {0}, {0}, {1} } ,
            });
        }

        public static Image<Arthmetic, double> Vector(params double[] vs)
        {
            var x = new Image<Arthmetic, double>(1, vs.Length);
            for(int i = 0; i < vs.Length; ++i)
            {
                x[i, 0] = vs[i];
            }
            return x;
        }
    }
}
