﻿using Egomotion;
using Emgu.CV;
using Emgu.CV.Util;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

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
                { {6.0}} ,
                { {6.0}} ,
                { {-3.0}} ,
            });

            //C2 = new Image<Arthmetic, double>(new double[,,] {
            //    { {0.0}} ,
            //    { {0.0}} ,
            //    { {0.0}} ,
            //});

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
            R12 = Rx(5.0).Multiply(Rz(5.0));
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
                    { {rand.Next(100, 200)}} ,
                    { {rand.Next(100, 200)}} ,
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

            FindTransformation.DecomposeToRTAndTriangulate(pts1, pts2, K, E, out var RR, out var tt, out Image<Arthmetic, double> X);

            var tt1 = T12.Mul(1 / T12.Norm);
            var tt2 = tt.Mul(1 / tt.Norm);

            var KK = EstimateCameraFromImagePair.K(F, 700, 400);
            var EE = ComputeMatrix.E(F, KK);
            var svd2 = new Svd(EE);
        }

        float Noise(double stddev, Random rng)
        {
            return (float)MathNet.Numerics.Distributions.Normal.Sample(rng, 0.0, stddev);
        }
        
        [TestMethod]
        public void TestMatricesFromNoisedPoints()
        {
            List<Image<Arthmetic, double>> ptsReal = new List<Image<Arthmetic, double>>();
            List<PointF> pts1 = new List<PointF>();
            List<PointF> pts2 = new List<PointF>();
            List<PointF> pts1Ref = new List<PointF>();
            List<PointF> pts2Ref = new List<PointF>();

            Random rand = new Random(1003);
            double stddev = 3;
            for (int i = 0; i < 100; ++i)
            {
                var real = new Image<Arthmetic, double>(new double[,,] {
                    { {rand.Next(100, 200)}} ,
                    { {rand.Next(100, 200)}} ,
                    { {rand.Next(50, 100)}} ,
                    { {1}} ,
                });

                ptsReal.Add(real);
                var i1 = P1.Multiply(real).ToPointF();
                pts1Ref.Add(i1);
                i1 = new PointF(i1.X + Noise(stddev, rand), i1.Y + Noise(stddev, rand));
                pts1.Add(i1);

                var i2 = P2.Multiply(real).ToPointF();
                pts2Ref.Add(i2);
                i2 = new PointF(i2.X + Noise(stddev, rand), i2.Y + Noise(stddev, rand));
                pts2.Add(i2);
            }

            double rangeLx = pts1.Max((x) => x.X) - pts1.Min((x) => x.X);
            double rangeLy = pts1.Max((x) => x.Y) - pts1.Min((x) => x.Y);
            double rangeRx = pts2.Max((x) => x.X) - pts2.Min((x) => x.X);
            double rangeRy = pts2.Max((x) => x.Y) - pts2.Min((x) => x.Y);

            var pts1_n = new List<PointF>(pts1);
            var pts2_n = new List<PointF>(pts2);
            FindTransformation.NormalizePoints2d(pts1_n, out Image<Arthmetic, double> NL);
            FindTransformation.NormalizePoints2d(pts2_n, out Image<Arthmetic, double> NR);

            MacthingResult match = new MacthingResult()
            {
                LeftPoints = new VectorOfPointF(pts1_n.ToArray()),
                RightPoints = new VectorOfPointF(pts2_n.ToArray()),
            };

            var F = ComputeMatrix.F(match.LeftPoints, match.RightPoints);

            // F is normalized - lets denormalize it
            F = NR.T().Multiply(F).Multiply(NL);

            var E = ComputeMatrix.E(F, K);

            var svd = new Svd(E);

            FindTransformation.DecomposeToRTAndTriangulate(pts1, pts2, K, E, out var RR, out var TT, out Image<Arthmetic, double> estReal);

            var rr0 = RotationConverter.MatrixToEulerXYZ(R12);
            var rr1 = RotationConverter.MatrixToEulerXYZ(RR);

            var tt0 = T12.Mul(1 / T12.Norm);
            var tt1 = TT.Mul(1 / TT.Norm);

            Errors.TraingulationError(ptsReal, estReal, out double mean1, out double median1, out List<double> errors1);
            Errors.ReprojectionError(estReal, pts2, K, RR, tt1, out double mean_r1a, out double median_r1a, out List<double> _1);
            Errors.ReprojectionError(estReal, pts2Ref, K, RR, tt1, out double mean_r1b, out double median_r1b, out List<double> _2);
            Errors.ReprojectionError(estReal, pts2Ref, K, R12, tt0, out double mean_r1c, out double median_r1c, out List<double> _3);
            Errors.ReprojectionError(Errors.Matrixify(ptsReal), pts2Ref, K, RR, tt1, out double mean_r1e, out double median_r1e, out List<double> _5);
            
            var H1 = FindTransformation.EstimateHomography(pts1, pts2, K);
            var H2 = FindTransformation.EstimateHomography(pts1Ref, pts2Ref, K);
            var hrr1 = RotationConverter.MatrixToEulerXYZ(H1);
            var hrr2 = RotationConverter.MatrixToEulerXYZ(H2);
            var zeroT = new Image<Arthmetic, double>(1, 3);
            
            var H3 = RotationConverter.EulerXYZToMatrix(hrr1);
            var hrr3 = RotationConverter.MatrixToEulerXYZ(H1);

            var svdH = new Svd(H1);

            bool isRotation = FindTransformation.IsPureRotation(H1);

            Errors.ReprojectionError2d(pts1Ref, pts2Ref, K, H2, out double mean_h2, out double median_h2, out var err_h2);
            Errors.ReprojectionError2d(pts1, pts2, K, H1, out double mean_h1, out double median_h1, out var err_h1);
            Errors.ReprojectionError2d(pts1, pts2, K, H3, out double mean_h3, out double median_h3, out var err_h3);

            Errors.ReprojectionError2dWithT(pts1, pts2, K, H1, zeroT, out double scale1, out double mean_h1a, out double median_h1a, out var err_h1a);
            Errors.ReprojectionError2dWithT(pts1, pts2, K, H3, zeroT, out double scale1x, out double mean_h1ax, out double median_h1ax, out var err_h1ax);
            //  Errors.ReprojectionError2dWithT(pts1, pts2, K, R12, tt0, out double scale2, out double mean_h1b, out double median_h1b, out var err_h1b);
            Errors.ReprojectionError2dWithT(pts1, pts2, K, RR, tt1, out double scale3, out double mean_h1c, out double median_h1c, out var err_h1c);
            Errors.ReprojectionError2dWithT(pts1, pts2, K, R12, tt1, out double scale5, out double mean_h1c1, out double median_h1c1, out var err_h1c1);
            Errors.ReprojectionError2dWithT(pts1Ref, pts2Ref, K, R12, tt0, out double scale6, out double mean_h1c2, out double median_h1c2, out var err_h1c2);
            Errors.ReprojectionError2dWithT(pts1, pts2, K, H1, tt1, out double scale4, out double mean_h1d, out double median_h1d, out var err_h1d);

            var KK = EstimateCameraFromImagePair.K(F, 600, 500);
            var EE = ComputeMatrix.E(F, KK);
            var svd2 = new Svd(EE);

            FindTransformation.DecomposeToRTAndTriangulate(pts1, pts2, KK, EE, out var RR2, out var TT2, out Image<Arthmetic, double> estReal2);
            var tt2 = TT2.Mul(1 / TT2.Norm);
            var rr2 = RotationConverter.MatrixToEulerXYZ(RR2);

            Errors.TraingulationError(ptsReal, estReal2, out double mean2, out double median2, out List<double> errors2);
            Errors.ReprojectionError(estReal2, pts2, KK, RR2, tt2, out double mean_r2a, out double median_r2a, out List<double> _1x);
            Errors.ReprojectionError(estReal2, pts2Ref, KK, RR2, tt2, out double mean_r2b, out double median_r2b, out List<double> _2x);
            Errors.ReprojectionError(estReal2, pts2Ref, KK, R12, tt0, out double mean_r2c, out double median_r2c, out List<double> _3x);
            Errors.ReprojectionError(Errors.Matrixify(ptsReal), pts2Ref, KK, RR2, tt2, out double mean_r2e, out double median_r2e, out List<double> _5x);
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
