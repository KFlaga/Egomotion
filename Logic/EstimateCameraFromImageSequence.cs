using Emgu.CV;
using Emgu.CV.Features2D;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Optimization;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Egomotion
{
    public class EstimateCameraFromImageSequence
    {
        public static Image<Arthmetic, double> K(List<Mat> mats, Feature2D detector, Feature2D descriptor, DistanceType distanceType, double maxDistance)
        {
            List<Image<Arthmetic, double>> Fs = new List<Image<Arthmetic, double>>();
            for (int i = 0; i < mats.Count - 1; i += 2)
            {
                var match = MatchImagePair.Match(mats[i], mats[i + 1], detector, descriptor, distanceType, maxDistance);
                var F = ComputeMatrix.F(match.LeftPoints, match.RightPoints);
                if (F == null)
                {
                    continue;
                }
                Fs.Add(F);
            }
            return K(Fs, mats[0].Width, mats[0].Height);
        }

        public static Image<Arthmetic, double> K(List<Image<Arthmetic, double>> Fs, double width, double height)
        {
            double fi = (width + height) / 2;
            var minimizer = new BfgsMinimizer(1e-6, 1e-6, 1e-6);
            var result = minimizer.FindMinimum(
                new ObjFunc() { Fs = Fs, Width = width, Height = height },
                new DenseVector(new double[] { fi, fi, width / 2, height / 2 })
            );
            var p = result.MinimizingPoint;
            return ComputeMatrix.K(p[0], p[1], p[2], p[3]);
        }

        public class ObjFunc : IObjectiveFunction
        {
            public List<Image<Arthmetic, double>> Fs { get; set; }
            public double Width { get; set; }
            public double Height { get; set; }

            public Vector<double> Point { get; set; }

            public double Value { get; set; }
            public bool IsGradientSupported => true;
            public Vector<double> Gradient { get; set; }
            public bool IsHessianSupported => false;
            public MathNet.Numerics.LinearAlgebra.Matrix<double> Hessian => throw new NotImplementedException();


            public void EvaluateAt(Vector<double> point)
            {
                Point = point;
                Value = Cost(Fs, point[0], point[1], point[2], point[3], Width, Height);
                Gradient = grad(point);
            }

            private Vector<double> grad(Vector<double> point)
            {
                Vector<double> g = new DenseVector(point.Count);
                for (int i = 0; i < point.Count; ++i)
                {
                    double hh = 1e-3;
                    Vector<double> p1 = point.Clone();
                    p1[i] = point[i] - hh;
                    double v1 = Cost(Fs, p1[0], p1[1], p1[2], p1[3], Width, Height);
                    p1[i] = point[i] + hh;
                    double v2 = Cost(Fs, p1[0], p1[1], p1[2], p1[3], Width, Height);
                    g[i] = (v2 - v1) / (2 * hh);
                }
                return g;
            }

            public static double Cost(List<Image<Arthmetic, double>> Fs, double fx, double fy, double px, double py, double w, double h)
            {
                double errS = 0;
                foreach(var F in Fs)
                {
                    Image<Arthmetic, double> E = ComputeMatrix.E(F, ComputeMatrix.K(fx, fy, px, py));
                    Svd svd = new Svd(E);

                    double s1 = svd.S[0, 0];
                    double s2 = svd.S[1, 0];
                    errS += s2 == 0 ? 1 : (s1 - s2) / s2;
                }
                return errS;
            }

            public IObjectiveFunction Fork()
            {
                return new ObjFunc() { Fs = Fs, Point = Point, Value = Value, Width = Width, Height = Height };
            }
            public IObjectiveFunction CreateNew()
            {
                return new ObjFunc() { Fs = Fs, Width = Width, Height = Height };
            }
        }

    }
}
