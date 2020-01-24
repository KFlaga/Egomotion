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

            K = new Image<Arthmetic, double>(new double[,,] {
                { {fx}, {0}, {px} } ,
                { {0}, {fy}, {py} } ,
                { {0}, {0 }, {1 } } ,
            });
            
            C2 = Utils.Vector(16, 16, -13);
            C3 = Utils.Vector(40, -20, 15);

            R12 = Utils.Rx(0.0).Multiply(Utils.Rz(5.0));
            R13 = Utils.Rx(0.0).Multiply(Utils.Rz(10.0));
            R23 = R12.T().Multiply(R13);
            
            C12 = C2.Clone();
            C23 = R12.Multiply(C3.Sub(C2));

            T12 = R12.Multiply(C2).Mul(-1);
            T13 = R13.Multiply(C3).Mul(-1);
            T23 = R23.Multiply(C23).Mul(-1);

            P1 = ComputeMatrix.Camera(K);
            P2 = ComputeMatrix.Camera(K, R12, C2);
            P3 = ComputeMatrix.Camera(K, R13, C3);

            Random rand = new Random(1003);
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
            pixelRange = 0.5 * (rangeLx + rangeLy + rangeRx + rangeRy);
        }

        private void ApplyNoise(double stddev)
        {
            pts1 = new List<PointF>();
            pts2 = new List<PointF>();
            pts3 = new List<PointF>();

            Random rand = new Random();
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

        void PlotInErrorFunction(Func<double, double> funcUnderTest)
        {
            List<double> points = new List<double>();
            double[] stddevs = new double[] { 0, 0.0005, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01 };
            foreach(double s in stddevs)
            {
                double scaledDev = s * pixelRange;
                points.Add(funcUnderTest(scaledDev));
            }

            axisY.Title = "StdDev";
            resetPlot();
            for (int i = 0; i < stddevs.Length; ++i)
            {
                series1.Points.Add(new DataPoint(stddevs[i], points[i]));
            }
            model.InvalidatePlot(true);
        }

        private void EigenRatioForKnownK_Click(object sender, RoutedEventArgs e)
        {
            PlotInErrorFunction((stddev) =>
            {
                ApplyNoise(stddev);

                var F = ComputeMatrix.F(new VectorOfPointF(pts1_n.ToArray()), new VectorOfPointF(pts2_n.ToArray()));
                // F is normalized - lets denormalize it
                F = N2.T().Multiply(F).Multiply(N1);

                var E = ComputeMatrix.E(F, K);
                var svd = new Svd(E);
                double error = svd.S[1, 0] / svd.S[0, 0];
                return error;
            });
        }

        private void ErrorOfKComputation_Click(object sender, RoutedEventArgs e)
        {

        }
    }
}
