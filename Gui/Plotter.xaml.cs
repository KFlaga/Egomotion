using Emgu.CV;
using Emgu.CV.Util;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace Egomotion
{
    public partial class Plotter : UserControl
    {
        // Plot data

        private PlotModel model = new PlotModel();
        private LinearAxis axisX = new LinearAxis() { Position = AxisPosition.Bottom };
        private LinearAxis axisY = new LinearAxis() { Position = AxisPosition.Left };
        
        private LineSeries series1 = new LineSeries()
        {
            StrokeThickness = 1,
            Color = OxyColor.FromRgb(0, 0, 0)
        };
        private LineSeries series2 = new LineSeries()
        {
            StrokeThickness = 1,
            Color = OxyColor.FromRgb(200, 0, 0)
        };
        private LineSeries series3 = new LineSeries()
        {
            StrokeThickness = 1,
            Color = OxyColor.FromRgb(0, 200, 0)
        };
        private LineSeries series4 = new LineSeries()
        {
            StrokeThickness = 1,
            Color = OxyColor.FromRgb(0, 0, 200)
        };
        private LineSeries[] allSeries => new LineSeries[] { series1, series2, series3, series4 };

        void resetPlot()
        {
            series1.Points.Clear();
            series2.Points.Clear();
            series3.Points.Clear();
            series4.Points.Clear();

            axisX.Title = "";
            axisY.Title = "";

            model.InvalidatePlot(true);
        }

        // Test data:

        double fx = 300;
        double fy = 250;
        double px = 320;
        double py = 240;

        Image<Arthmetic, double> K;

        Image<Arthmetic, double> C2;
        Image<Arthmetic, double> C3;
        Image<Arthmetic, double> C12;
        Image<Arthmetic, double> C23;

        Image<Arthmetic, double> R12;
        Image<Arthmetic, double> R13;
        Image<Arthmetic, double> R23;

        Image<Arthmetic, double> T12;
        Image<Arthmetic, double> T13;
        Image<Arthmetic, double> T23;

        Image<Arthmetic, double> P1;
        Image<Arthmetic, double> P2;
        Image<Arthmetic, double> P3;

        List<Image<Arthmetic, double>> ptsReal = new List<Image<Arthmetic, double>>();

        List<PointF> pts1Ref = new List<PointF>();
        List<PointF> pts2Ref = new List<PointF>();
        List<PointF> pts3Ref = new List<PointF>();

        // Noised points
        List<PointF> pts1 = new List<PointF>();
        List<PointF> pts2 = new List<PointF>();
        List<PointF> pts3 = new List<PointF>();

        // Normalized noised points
        List<PointF> pts1_n = new List<PointF>();
        List<PointF> pts2_n = new List<PointF>();
        List<PointF> pts3_n = new List<PointF>();

        // Normalization metrices
        Image<Arthmetic, double> N1;
        Image<Arthmetic, double> N2;
        Image<Arthmetic, double> N3;

        int pointsCount = 100;

        double pixelRange = 0.0;

        public Plotter()
        {
            InitializeComponent();

            model.Axes.Add(axisX);
            model.Axes.Add(axisY);

            model.Series.Add(series1);
            model.Series.Add(series2);
            model.Series.Add(series3);
            model.Series.Add(series4);

            plot.Model = model;

            ResetTestMatrices();
            ResetPoints();
        }

        private void ResetTestMatrices()
        {
            if (!int.TryParse(pointCountInput.Text, out pointsCount) || pointsCount <= 8)
            {
                pointCountInput.Text = "100";
                pointsCount = 100;
            }

            K = new Image<Arthmetic, double>(new double[,,] {
                { {fx}, {0}, {px} } ,
                { {0}, {fy}, {py} } ,
                { {0}, {0 }, {1 } } ,
            });

            C2 = Utils.Vector(8, 8, -6);
            C3 = Utils.Vector(40, -20, 15);

            R12 = Utils.Rx(0.0).Multiply(Utils.Rz(5.0));
            R13 = Utils.Rx(0.0).Multiply(Utils.Rz(5.0));
            
            R23 = R13.Multiply(R12.T());

            C12 = C2.Clone();
            C23 = R12.Multiply(C3.Sub(C2));

            T12 = R12.Multiply(C2).Mul(-1);
            T13 = R13.Multiply(C3).Mul(-1);
            T23 = R23.Multiply(C23).Mul(-1);
        }

        private void ResetPoints()
        {
            P1 = ComputeMatrix.Camera(K);
            P2 = ComputeMatrix.Camera(K, R12, C2);
            P3 = ComputeMatrix.Camera(K, R13, C3);

            ptsReal = new List<Image<Arthmetic, double>>();
            pts1Ref = new List<PointF>();
            pts2Ref = new List<PointF>();
            pts3Ref = new List<PointF>();

            Random rand = new Random(564073);
            for (int i = 0; i < pointsCount; ++i)
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

                var i2 = P2.Multiply(real).ToPointF();
                pts2Ref.Add(i2);

                var i3 = P3.Multiply(real).ToPointF();
                pts3Ref.Add(i3);
            }

            double rangeLx = pts1Ref.Max((x) => x.X) - pts1Ref.Min((x) => x.X);
            double rangeLy = pts1Ref.Max((x) => x.Y) - pts1Ref.Min((x) => x.Y);
            double rangeRx = pts2Ref.Max((x) => x.X) - pts2Ref.Min((x) => x.X);
            double rangeRy = pts2Ref.Max((x) => x.Y) - pts2Ref.Min((x) => x.Y);
            pixelRange = 0.25 * (rangeLx + rangeLy + rangeRx + rangeRy);
        }

        int seed = 679447;
        private void ApplyNoise(double stddev)
        {
            pts1 = new List<PointF>();
            pts2 = new List<PointF>();
            pts3 = new List<PointF>();

            Random rand = new Random(seed);
            for (int i = 0; i < pointsCount; ++i)
            {
                var real = ptsReal[i];
                var i1 = pts1Ref[i];
                var i2 = pts2Ref[i];
                var i3 = pts3Ref[i];

                i1 = new PointF(i1.X + Noise(stddev, rand), i1.Y + Noise(stddev, rand));
                pts1.Add(i1);
                
                i2 = new PointF(i2.X + Noise(stddev, rand), i2.Y + Noise(stddev, rand));
                pts2.Add(i2);
                
                i3 = new PointF(i3.X + Noise(stddev, rand), i3.Y + Noise(stddev, rand));
                pts3.Add(i3);
            }

            pts1_n = new List<PointF>(pts1);
            pts2_n = new List<PointF>(pts2);
            pts3_n = new List<PointF>(pts3);
            FindTransformation.NormalizePoints2d(pts1_n, out N1);
            FindTransformation.NormalizePoints2d(pts2_n, out N2);
            FindTransformation.NormalizePoints2d(pts3_n, out N3);
        }

        float Noise(double stddev, Random rng)
        {
            return (float)MathNet.Numerics.Distributions.Normal.Sample(rng, 0.0, stddev);
        }

        public class PlotDefinition
        {
            public string YName;
            public Func<double> Function;

            public int CasesCount = 1;
            public List<Action<double>> PrepareCase;
            public List<string> CasesNames;
        }

        void PlotFunctionForErrors(Func<double> funcUnderTest, string yName)
        {
            PlotFunctionForErrors(new PlotDefinition()
            {
                Function = funcUnderTest,
                YName = yName,
                CasesCount = 1,
                CasesNames = new List<string>() { "" },
                PrepareCase = new List<Action<double>>() { (_) => { } }
            });
        }

        void PlotFunctionForErrors(PlotDefinition plotDef)
        {
            resetPlot();
            double[] stddevs = new double[] { 0, 0.00025, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.01 };

            for (int i = 0; i < plotDef.CasesCount; ++i)
            {
                ResetTestMatrices();
                ResetPoints();

                allSeries[i].Title = plotDef.CasesNames[i];

                List<double> points = new List<double>();
                foreach (double s in stddevs)
                {
                    double scaledDev = s * pixelRange;
                    plotDef.PrepareCase[i](scaledDev);
                    ApplyNoise(scaledDev);
                    points.Add(plotDef.Function());
                }

                for (int k = 0; k < stddevs.Length; ++k)
                {
                    allSeries[i].Points.Add(new DataPoint(stddevs[k] * 100.0, points[k]));
                }
            }

            axisX.Title = "StdDev [%]";
            axisY.Title = plotDef.YName;
            model.InvalidatePlot(true);
        }

        private void Reseed_Click(object sender, RoutedEventArgs e)
        {
            seed = new Random().Next(1001, 999999);
        }

        private Image<Arthmetic, double> ComputeF()
        {
            var F = ComputeMatrix.F(new VectorOfPointF(pts1_n.ToArray()), new VectorOfPointF(pts2_n.ToArray()));
            // F is normalized - lets denormalize it
            F = N2.T().Multiply(F).Multiply(N1);
            return F;
        }

        private void ErrorOfFComputation_Click(object sender, RoutedEventArgs e)
        {
            ResetTestMatrices();
            ResetPoints();
            ApplyNoise(0);
            var Fref = ComputeF();
            Fref = Fref.Mul(1.0 / Fref[2, 2]);

            PlotFunctionForErrors(() =>
            {
                var F = ComputeF();
                F = F.Mul(1.0 / F[2, 2]);
                double error = (F - Fref).Norm;
                return error;
            }, "||Fref - Fest||");
        }

        private void EigenRatioForKnownK_Click(object sender, RoutedEventArgs e)
        {
            PlotFunctionForErrors(() =>
            {
                var F = ComputeF();
                var E = ComputeMatrix.E(F, K);
                var svd = new Svd(E);
                double error = svd.S[1, 0] / svd.S[0, 0];

                return error;
            }, "s2 / s1");
        }

        private void ErrorOfKComputation_Click(object sender, RoutedEventArgs e)
        {
            PlotFunctionForErrors(() =>
            {
                var F = ComputeF();
                var Kest = EstimateCameraFromImagePair.K(F, px * 2, py * 2);
                double error = (Kest - K).Norm / (fx + fy);

                return error;
            }, "|fx - fy_est|+|fy - fy_est| / (fx + fy)");
        }

        private void ErrorOfRComputation_Click(object sender, RoutedEventArgs e)
        {
            Func<double> testFunc = () =>
            {
                var F = ComputeF();
                var E = ComputeMatrix.E(F, K);
                FindTransformation.DecomposeToRTAndTriangulate(pts1, pts2, K, E, out var R, out var t, out var pts3d);
                return FindRError(R);
            };
            PrepareKs(out var prepareFunc, out var casesNames);
            PlotFunctionForErrors(new PlotDefinition()
            {
                YName = "||angles - angles_ref||",
                Function = testFunc,
                CasesCount = 2,
                PrepareCase = prepareFunc,
                CasesNames = casesNames
            });
        }

        private double FindRError(Image<Arthmetic, double> R)
        {
            var rEst = RotationConverter.MatrixToEulerXYZ(R);
            var rRef = RotationConverter.MatrixToEulerXYZ(R12);
            var diff = (rEst - rRef);
            double error = 0.0;
            for (int i = 0; i < 3; ++i)
            {
                error += Math.Abs(diff[i, 0]);
            }
            return error * 180.0 / Math.PI;
        }

        private void PrepareCs(out List<Action<double>> prepareFunc, out List<string> casesNames)
        {
            prepareFunc = new List<Action<double>>()
            {
                (_) => { C2 = Utils.Vector(0, 0, 0); C3 = C2.Mul(2.0); ResetPoints(); },
                (_) => { C2 = Utils.Vector(0.57, 0.57, 0.57); C3 = C2.Mul(2.0); ResetPoints(); },
                (_) => { C2 = Utils.Vector(2.88, 2.88, 2.88); C3 = C2.Mul(2.0); ResetPoints(); },
                (_) => { C2 = Utils.Vector(14.4, 14.4, 14.4); C3 = C2.Mul(2.0); ResetPoints(); },
            };
            casesNames = new List<string>() { "||C|| = 0", "||C|| = 1", "||C|| = 5", "||C|| = 25" };
        }

        private void ErrorOfRComputation2_Click(object sender, RoutedEventArgs e)
        {
            Func<double> testFunc = () =>
            {
                var F = ComputeF();
                var E = ComputeMatrix.E(F, K);
                FindTransformation.DecomposeToRTAndTriangulate(pts1, pts2, K, E, out var R, out var t, out var pts3d);
                return FindRError(R);
            };
            PrepareCs(out var prepareFunc, out var casesNames);
            PlotFunctionForErrors(new PlotDefinition()
            {
                YName = "||angles - angles_ref||",
                Function = testFunc,
                CasesCount = 4,
                PrepareCase = prepareFunc,
                CasesNames = casesNames
            });
        }

        private void ErrorOfHComputation_Click(object sender, RoutedEventArgs e)
        {
            Func<double> testFunc = () =>
            {
                var H = FindTransformation.EstimateHomography(pts1, pts2, K);
                var svdH = new Svd(H);
                return FindRError(H);
            };
            PrepareCs(out var prepareFunc, out var casesNames);
            PlotFunctionForErrors(new PlotDefinition()
            {
                YName = "||angles - angles_ref||",
                Function = testFunc,
                CasesCount = 4,
                PrepareCase = prepareFunc,
                CasesNames = casesNames
            });
        }

        private void PrepareKs(out List<Action<double>> prepareFunc, out List<string> casesNames)
        {
            prepareFunc = new List<Action<double>>()
            {
                (_) => {},
                (stddev) =>
                {
                    ResetPoints();
                    ApplyNoise(stddev);
                    var F = ComputeF();
                    K = EstimateCameraFromImagePair.K(F, px * 2, py * 2);
                }
            };
            casesNames = new List<string>() { "Kref", "Kest" };
        }

        private double FindTError(Image<Arthmetic, double> t_est)
        {
            if (C2.Norm < 1e-3)
            {
                return 1.0;
            }

            var t_ref = R12.Multiply(C2).Mul(-1);
            t_ref = t_ref.Mul(1.0 / t_ref.Norm);
            t_est = t_est.Mul(1.0 / t_est.Norm);
            return Math.Min((t_ref - t_est).Norm, (t_ref + t_est).Norm); // check + and - as it may be scaled by -1 which is ok
        }

        private void ErrorOfTComputation_Click(object sender, RoutedEventArgs e)
        {
            Func<double> testFunc = () =>
            {
                var F = ComputeF();
                var E = ComputeMatrix.E(F, K);
                FindTransformation.DecomposeToRTAndTriangulate(pts1, pts2, K, E, out var R, out var t, out var pts3d);
                return FindTError(t);
            };
            PrepareKs(out var prepareFunc, out var casesNames);
            PlotFunctionForErrors(new PlotDefinition()
            {
                YName = "||t_ref - t_est||",
                Function = testFunc,
                CasesCount = 2,
                PrepareCase = prepareFunc,
                CasesNames = casesNames
            });
        }

        private void ErrorOfTComputation2_Click(object sender, RoutedEventArgs e)
        {
            Func<double> testFunc = () =>
            {
                var F = ComputeF();
                var E = ComputeMatrix.E(F, K);
                FindTransformation.DecomposeToRTAndTriangulate(pts1, pts2, K, E, out var R, out var t, out var pts3d);
                return FindTError(t);
            };
            PrepareCs(out var prepareFunc, out var casesNames);
            PlotFunctionForErrors(new PlotDefinition()
            {
                YName = "||t_ref - t_est||",
                Function = testFunc,
                CasesCount = 4,
                PrepareCase = prepareFunc,
                CasesNames = casesNames
            });
        }

        private void ErrorOfHComputation2_Click(object sender, RoutedEventArgs e)
        {
            Func<double> testFunc = () =>
            {
                var H = FindTransformation.EstimateHomography(pts1, pts2, K);
                var svdH = new Svd(H);
                return svdH.S[2, 0] / svdH.S[0, 0];
            };
            PrepareCs(out var prepareFunc, out var casesNames);
            PlotFunctionForErrors(new PlotDefinition()
            {
                YName = "s3 / s1",
                Function = testFunc,
                CasesCount = 4,
                PrepareCase = prepareFunc,
                CasesNames = casesNames
            });
        }

        private void ErrorOfHComputation3_Click(object sender, RoutedEventArgs e)
        {
            Func<double> testFunc = () =>
            {
                var H = FindTransformation.EstimateHomography(pts1, pts2, K);
                var svdH = new Svd(H);
                return Math.Abs(svdH.S[0, 0] - 1);
            };
            PrepareCs(out var prepareFunc, out var casesNames);
            PlotFunctionForErrors(new PlotDefinition()
            {
                YName = "|s1 - 1|",
                Function = testFunc,
                CasesCount = 4,
                PrepareCase = prepareFunc,
                CasesNames = casesNames
            });
        }

        private void ErrorOfScaleComputation_Click(object sender, RoutedEventArgs e)
        {
            Func<double> testFunc = () =>
            {
                if (C2.Norm < 1e-3)
                {
                    return 1.0;
                }

                var F12 = ComputeMatrix.F(new VectorOfPointF(pts1_n.ToArray()), new VectorOfPointF(pts2_n.ToArray()));
                F12 = N2.T().Multiply(F12).Multiply(N1);

                var F23 = ComputeMatrix.F(new VectorOfPointF(pts2_n.ToArray()), new VectorOfPointF(pts3_n.ToArray()));
                F23 = N3.T().Multiply(F23).Multiply(N2);

                var E12 = ComputeMatrix.E(F12, K);
                var E23 = ComputeMatrix.E(F23, K);

                FindTransformation.DecomposeToRTAndTriangulate(pts1, pts2, K, E12, out var eR12, out var eT12, out var pts3d12);
                FindTransformation.DecomposeToRTAndTriangulate(pts2, pts3, K, E23, out var eR23, out var eT23, out var pts3d23);

                eT12 = eT12.Mul(1.0 / eT12.Norm);
                eT23 = eT23.Mul(1.0 / eT23.Norm);

                ScaleBy3dPointsMatch.FindBestScale(eR12, eT12, eR23, eT23, K, pts1, pts2, pts3, 10, out double scale, out double confidence, out var inliners);
                
                double refScale = eT23.Norm / eT12.Norm;
                double error = scale / refScale;

                return error;
            };
            PrepareCs(out var prepareFunc, out var casesNames);
            PlotFunctionForErrors(new PlotDefinition()
            {
                YName = "scale_est / scale_ref; scale = ||t23|| / ||t12||",
                Function = testFunc,
                CasesCount = 4,
                PrepareCase = prepareFunc,
                CasesNames = casesNames
            });
        }

        private void ErrorOfScaleComputation2_Click(object sender, RoutedEventArgs e)
        {
            Func<double> testFunc = () =>
            {
                if (C2.Norm < 1e-3)
                {
                    return 1.0;
                }

                C12 = C2.Clone();
                C23 = R12.Multiply(C3.Sub(C2));

                T12 = R12.Multiply(C12).Mul(-1);
                T23 = R23.Multiply(C23).Mul(-1);

                //T12 = T12.Mul(1.0 / T12.Norm);
                //if (T12[0, 0] < 0)
                //    T12 = T12.Mul(-1.0);

                //T23 = T23.Mul(1.0 / T23.Norm);
                //if (T23[0, 0] < 0)
                //    T23 = T23.Mul(-1.0);

                FindTransformation.TriangulateChieral(pts1, pts2, K, R12, T12, out var est3d_12);
                FindTransformation.TriangulateChieral(pts2, pts3, K, R23, T23, out var est3d_23);
                var backprojected23to12 = ScaleBy3dPointsMatch.TransfromBack3dPoints(R12, T12, est3d_23, 1.0);

                ScaleBy3dPointsMatch.FindBestScale(R12, T12, R23, T23, K, pts1, pts2, pts3, 10, out double scale, out double confidence, out var inliners);
                
                double refScale = T23.Norm / T12.Norm;
                double error = scale / refScale;

                return error;
            };
            PrepareCs(out var prepareFunc, out var casesNames);
            PlotFunctionForErrors(new PlotDefinition()
            {
                YName = "scale_est / scale_ref; scale = ||t23|| / ||t12||",
                Function = testFunc,
                CasesCount = 4,
                PrepareCase = prepareFunc,
                CasesNames = casesNames
            });
        }
    }
}
