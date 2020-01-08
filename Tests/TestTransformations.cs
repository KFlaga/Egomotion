using Egomotion;
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

        void normalizeFeaturePoints(List<PointF> left, List<PointF> right, out Image<Arthmetic, double>  NL, out Image<Arthmetic, double>  NR)
        {
            // Compute centroid of both point sets
            float mean_lx = 0, mean_ly = 0, mean_rx = 0, mean_ry = 0;
            for (int i = 0; i < left.Count; ++i)
            {
                mean_lx += left[i].X;
                mean_ly += left[i].Y;
                mean_rx += right[i].X;
                mean_ry += right[i].Y;
            }
            mean_lx /= left.Count;
            mean_ly /= left.Count;
            mean_rx /= left.Count;
            mean_ry /= left.Count;

            // Shift origins to centroids
            for (int i = 0; i < left.Count; ++i)
            {
                left[i] = PointF.Subtract(left[i], new SizeF(mean_lx, mean_ly));
                right[i] = PointF.Subtract(right[i], new SizeF(mean_rx, mean_ry));
            }

            // Scale points so that mean distance from origin is sqrt(2)
            float scale_l = 0, scale_r = 0;
            for (int i = 0; i < left.Count; ++i)
            {
                scale_l += (float)Math.Sqrt(left[i].X * left[i].X + left[i].Y * left[i].Y);
                scale_r += (float)Math.Sqrt(right[i].X * right[i].X + right[i].Y * right[i].Y);
            }

            float targetMean = (float)Math.Sqrt(2.0);
            scale_l = targetMean * left.Count / scale_l;
            scale_r = targetMean * left.Count / scale_r;
            for (int i = 0; i < left.Count; ++i)
            {
                left[i] = new PointF(left[i].X * scale_l, left[i].Y * scale_l);
                right[i] = new PointF(right[i].X * scale_r, right[i].Y * scale_r);
            }

            // compute corresponding transformation matrices
            NL = new Image<Arthmetic, double>(new double[,,] {
                { {scale_l}, {0}, {-scale_l * mean_lx}, } ,
                { {0}, {scale_l}, {-scale_l * mean_ly}, } ,
                { {0}, {0}, {1}, } ,
            });

            NR = new Image<Arthmetic, double>(new double[,,] {
                { {scale_r}, {0}, {-scale_r * mean_rx}, } ,
                { {0}, {scale_r}, {-scale_r * mean_ry}, } ,
                { {0}, {0}, {1}, } ,
            });
        }
        
        [TestMethod]
        public void TestMatricesFromNoisedPoints()
        {
            List<Image<Arthmetic, double>> ptsReal = new List<Image<Arthmetic, double>>();
            List<PointF> pts1 = new List<PointF>();
            List<PointF> pts2 = new List<PointF>();

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
                i1 = new PointF(i1.X + Noise(stddev, rand), i1.Y + Noise(stddev, rand));
                pts1.Add(i1);
                
                var i2 = P2.Multiply(real).ToPointF();
                i2 = new PointF(i2.X + Noise(stddev, rand), i2.Y + Noise(stddev, rand));
                pts2.Add(i2);
            }

            double rangeLx = pts1.Max((x) => x.X) - pts1.Min((x) => x.X);
            double rangeLy = pts1.Max((x) => x.Y) - pts1.Min((x) => x.Y);
            double rangeRx = pts2.Max((x) => x.X) - pts2.Min((x) => x.X);
            double rangeRy = pts2.Max((x) => x.Y) - pts2.Min((x) => x.Y);

            var pts1_n = new List<PointF>(pts1);
            var pts2_n = new List<PointF>(pts2);
            normalizeFeaturePoints(pts1_n, pts2_n, out Image<Arthmetic, double> NL, out Image<Arthmetic, double> NR);

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
            
            List<double> errors = new List<double>();
            for(int i = 0; i < pts1.Count; ++i)
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

            var KK = EstimateCameraFromImagePair.K(F, 640, 480);
            var EE = ComputeMatrix.E(F, KK);
            var svd2 = new Svd(EE);

            FindTransformation.DecomposeToRTAndTriangulate(pts1, pts2, KK, EE, out var RR2, out var TT2, out Image<Arthmetic, double> estReal2);
            var tt2 = TT2.Mul(1 / TT2.Norm);
            var rr2 = RotationConverter.MatrixToEulerXYZ(RR2);
            
            List<double> errors2 = new List<double>();
            for (int i = 0; i < pts1.Count; ++i)
            {
                var estPoint = new Image<Arthmetic, double>(new double[,,]
                {
                    {{estReal2[0, i]}}, {{estReal2[1, i]}}, {{estReal2[2, i]}},
                });
                var realPoint = new Image<Arthmetic, double>(new double[,,]
                {
                    {{ptsReal[i][0, 0]}}, {{ptsReal[i][1, 0]}}, {{ptsReal[i][2, 0]}},
                });

                var p1 = estPoint.Mul(1 / estPoint.Norm);
                var p2 = realPoint.Mul(1 / realPoint.Norm);

                errors2.Add(p1.Sub(p2).Norm);
            }
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
