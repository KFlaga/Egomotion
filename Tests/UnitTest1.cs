using Egomotion;
using Emgu.CV;
using Emgu.CV.Util;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Drawing;

namespace Tests
{
    [TestClass]
    public class UnitTest1
    {
        [TestMethod]
        public void TestIdeal()
        {
            double fx = 300;
            double fy = 250;
            double px = 320;
            double py = 240;

            var K = new Image<Arthmetic, double>(new double[,,] {
                { {fx}, {0}, {px} } ,
                { {0}, {fy}, {py} } ,
                { {0}, {0 }, {1 } } ,
            });

            var C = new Image<Arthmetic, double>(new double[,,] {
                { {50.0}} ,
                { {30.0}} ,
                { {-20.0}} ,
            });
            
            var R = new Image<Arthmetic, double>(new double[,,] {
                { {1}, {0}, {0} } ,
                { {0}, {1}, {0} } ,
                { {0}, {0}, {1} } ,
            });

            var T = R.Multiply(C).Mul(-1);

            var P1 = new Image<Arthmetic, double>(new double[,,] {
                { {fx}, {0}, {px}, {0} } ,
                { {0}, {fy}, {py}, {0} } ,
                { {0}, {0}, {1}, {0} } ,
            });

            var P2 = new Image<Arthmetic, double>(new double[,,] {
                { {1}, {0}, {0}, {-C[0, 0]} } ,
                { {0}, {1}, {0}, {-C[1, 0]} } ,
                { {0}, {0}, {1}, {-C[2, 0]} } ,
            });
            P2 = K.Multiply(R).Multiply(P2);

            List<Image<Arthmetic, double>> ptsReal = new List<Image<Arthmetic, double>>();
            List<PointF> pts1 = new List<PointF>();
            List<PointF> pts2 = new List<PointF>();

            Random rand = new Random(1001);
            for(int i = 0; i < 10; ++i)
            {
                var real = new Image<Arthmetic, double>(new double[,,] {
                    { {rand.Next(-100, 100)}} ,
                    { {rand.Next(-100, 100)}} ,
                    { {rand.Next(50, 100)}} ,
                    { {1}} ,
                });

                ptsReal.Add(real);
                var i1 = P1.Multiply(real).ToPointF();
                pts1.Add(i1);

                var i2 = P2.Multiply(real).ToPointF();
                pts2.Add(i2);
            }

            MacthingResult match = new MacthingResult()
            {
                LeftPoints = new VectorOfPointF(pts1.ToArray()),
                RightPoints = new VectorOfPointF(pts2.ToArray()),
            };

            var F = ComputeMatrix.F(match.LeftPoints, match.RightPoints);
            var E = ComputeMatrix.E(F, K);
            var svd = new Svd(E);

            FindTransformation.DecomposeToRT(E, out var RR, out var TT);
            var tt = ComputeMatrix.CrossProductToVector(TT);

            var tt1 = T.Mul(1 / T.Norm);
            var tt2 = tt.Mul(1 / tt.Norm);

            var CC = FindTransformation.ComputeCameraCenter(R, T, K, match, 1.0);
        }
    }
}
