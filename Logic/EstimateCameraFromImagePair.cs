﻿using Emgu.CV;
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
    // Assumes (px, py) = (w/2, h/2), this gives better results than using all 4 parameters
    public class EstimateCameraFromImagePair
    {
        public static Image<Arthmetic, double> K(Image<Arthmetic, double> F, double width, double height)
        {
            double fi = (width + height) / 2;
            var minimizer = new BfgsMinimizer(1e-6, 1e-6, 1e-6);
            var result = minimizer.FindMinimum(
                new ObjFunc() { F = F, Width = width, Height = height },
                new DenseVector(new double[] { fi, fi })
            );
            var p = result.MinimizingPoint;
            return ComputeMatrix.K(p[0], p[1], width / 2, height / 2);
        }

        public class ObjFunc : IObjectiveFunction
        {
            public Image<Arthmetic, double> F { get; set; }
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
                Value = Cost(F, point[0], point[1], Width / 2, Height / 2, Width, Height);
                Gradient = Grad(point);
            }

            public Vector<double> Grad(Vector<double> point)
            {
                Vector<double> g = new DenseVector(point.Count);
                for (int i = 0; i < point.Count; ++i)
                {
                    double hh = 1e-3;
                    Vector<double> p1 = point.Clone();
                    p1[i] = point[i] - hh;
                    double v1 = Cost(F, p1[0], p1[1], Width / 2, Height / 2, Width, Height);
                    p1[i] = point[i] + hh;
                    double v2 = Cost(F, p1[0], p1[1], Width / 2, Height / 2, Width, Height);
                    g[i] = (v2 - v1) / (2 * hh);
                }
                return g;
            }

            public double Cost(Image<Arthmetic, double> F, double fx, double fy, double px, double py, double w, double h)
            {
                Image<Arthmetic, double> E = ComputeMatrix.E(F, ComputeMatrix.K(fx, fy, px, py));
                Svd svd = new Svd(E);

                double s1 = svd.S[0, 0];
                double s2 = svd.S[1, 0];
                double errS = s2 == 0 ? 1.0 : (s1 - s2) / s2;
                return errS;
            }

            public IObjectiveFunction Fork()
            {
                return new ObjFunc() { F = F, Point = Point, Value = Value, Width = Width, Height = Height };
            }

            public IObjectiveFunction CreateNew()
            {
                return new ObjFunc() { F = F, Width = Width, Height = Height };
            }
        }
    }
}
