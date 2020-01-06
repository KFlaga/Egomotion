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
    public class TestTransformations
    {
        double fx = 300;
        double fy = 250;
        double px = 320;
        double py = 240;
        Image<Arthmetic, double> K;
        Image<Arthmetic, double> C2;
        Image<Arthmetic, double> C3;
        Image<Arthmetic, double> C4;
        Image<Arthmetic, double> C23;
        Image<Arthmetic, double> C24;
        Image<Arthmetic, double> C34;
        Image<Arthmetic, double> R1;
        Image<Arthmetic, double> R12;
        Image<Arthmetic, double> R13;
        Image<Arthmetic, double> R14;
        Image<Arthmetic, double> R23;
        Image<Arthmetic, double> R24;
        Image<Arthmetic, double> R34;
        Image<Arthmetic, double> T12;
        Image<Arthmetic, double> T13;
        Image<Arthmetic, double> T14;
        Image<Arthmetic, double> T23;
        Image<Arthmetic, double> T24;
        Image<Arthmetic, double> T34;
        Image<Arthmetic, double> P1;
        Image<Arthmetic, double> P2;
        Image<Arthmetic, double> P3;
        Image<Arthmetic, double> P4;

        Image<Arthmetic, double> Rx(double a)
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

        Image<Arthmetic, double> Rz(double a)
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

        Image<Arthmetic, double> I()
        {
            return new Image<Arthmetic, double>(new double[,,] {
                { {1}, {0}, {0} } ,
                { {0}, {1}, {0} } ,
                { {0}, {0}, {1} } ,
            });
        }

        [TestInitialize]
        public void Init()
        {
            K = new Image<Arthmetic, double>(new double[,,] {
                { {fx}, {0}, {px} } ,
                { {0}, {fy}, {py} } ,
                { {0}, {0 }, {1 } } ,
            });

            C2 = new Image<Arthmetic, double>(new double[,,] {
                { {50.0}} ,
                { {30.0}} ,
                { {-20.0}} ,
            });

            C3 = new Image<Arthmetic, double>(new double[,,] {
                { {30.0}} ,
                { {20.0}} ,
                { {-30.0}} ,
            });

            C4 = new Image<Arthmetic, double>(new double[,,] {
                { {20.0}} ,
                { {30.0}} ,
                { {-10.0}} ,
            });

            R1 = I();
            R12 = Rz(10.0);
            R13 = Rz(25.0);
            R14 = Rz(15.0);
            R23 = Rz(15.0);
            R24 = Rz(5.0);
            R34 = Rz(-10.0);

            T12 = R12.Multiply(C2).Mul(-1);
            T13 = R13.Multiply(C3).Mul(-1);
            T14 = R14.Multiply(C4).Mul(-1);

            C23 = R12.Multiply(C3.Sub(C2));
            C24 = R12.Multiply(C4.Sub(C2));
            C34 = C4.Sub(C3);
            T23 = R23.Multiply(C23);
            T24 = R24.Multiply(C24);
            T34 = R34.Multiply(C34);

            P1 = new Image<Arthmetic, double>(new double[,,] {
                { {1}, {0}, {0}, {0} } ,
                { {0}, {1}, {0}, {0} } ,
                { {0}, {0}, {1}, {0} } ,
            });
            P1 = K.Multiply(P1);

            P2 = new Image<Arthmetic, double>(new double[,,] {
                { {1}, {0}, {0}, {-C2[0, 0]} } ,
                { {0}, {1}, {0}, {-C2[1, 0]} } ,
                { {0}, {0}, {1}, {-C2[2, 0]} } ,
            });
            P2 = K.Multiply(R12).Multiply(P2);

            P3 = new Image<Arthmetic, double>(new double[,,] {
                { {1}, {0}, {0}, {-C3[0, 0]} } ,
                { {0}, {1}, {0}, {-C3[1, 0]} } ,
                { {0}, {0}, {1}, {-C3[2, 0]} } ,
            });
            P3 = K.Multiply(R13).Multiply(P3);

            P4 = new Image<Arthmetic, double>(new double[,,] {
                { {1}, {0}, {0}, {-C4[0, 0]} } ,
                { {0}, {1}, {0}, {-C4[1, 0]} } ,
                { {0}, {0}, {1}, {-C4[2, 0]} } ,
            });
            P4 = K.Multiply(R14).Multiply(P4);
        }

        [TestMethod]
        public void TestIdealMatrices()
        {
            List<Image<Arthmetic, double>> ptsReal = new List<Image<Arthmetic, double>>();
            List<PointF> pts1 = new List<PointF>();
            List<PointF> pts2 = new List<PointF>();

            Random rand = new Random(1003);
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

            var tt1 = T12.Mul(1 / T12.Norm);
            var tt2 = tt.Mul(1 / tt.Norm);

            var KK = EstimateCameraFromImagePair.K(F, 700, 400);
            var EE = ComputeMatrix.E(F, KK);
            var svd2 = new Svd(EE);
        }

        [TestMethod]
        public void TestIdealTranslaionScaling()
        {
            List<Image<Arthmetic, double>> ptsReal = new List<Image<Arthmetic, double>>();
            List<PointF> pts1 = new List<PointF>();
            List<PointF> pts2 = new List<PointF>();
            List<PointF> pts3 = new List<PointF>();
            List<PointF> pts4 = new List<PointF>();

            Random rand = new Random(1003);
            for (int i = 0; i < 10; ++i)
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

                var i3 = P3.Multiply(real).ToPointF();
                pts3.Add(i3);

                var i4 = P4.Multiply(real).ToPointF();
                pts4.Add(i4);
            }

            TripletMatch match13 = new TripletMatch()
            {
                Left = pts1,
                Middle = pts2,
                Right = pts3,
            };

            TripletMatch match24 = new TripletMatch()
            {
                Left = pts2,
                Middle = pts3,
                Right = pts4,
            };

            List<Image<Arthmetic, double>> Rs1 = new List<Image<Arthmetic, double>>() { R12, R13 };
            List<Image<Arthmetic, double>> Ts1 = new List<Image<Arthmetic, double>>() { T12, T13 };
            
            List<Image<Arthmetic, double>> Rs2 = new List<Image<Arthmetic, double>>() { R23, R24 };
            List<Image<Arthmetic, double>> Ts2 = new List<Image<Arthmetic, double>>() { T23, T24 };
            
            var centers123 = FindTransformation.ComputeCameraCenter3(K, Rs1, Ts1, match13);
            var centers234 = FindTransformation.ComputeCameraCenter3(K, Rs2, Ts2, match24);

            var C12_1_abs = centers123.C12;
            var C13_1_abs = centers123.C13;

            var C23_2_abs = R12.T().Multiply(centers234.C12);
            var C24_2_abs = R12.T().Multiply(centers234.C13);
        }
    }
}
