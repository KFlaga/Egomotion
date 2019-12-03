using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.CV.Structure;

namespace Egomotion
{
    public struct Arthmetic : Emgu.CV.IColor, IEquatable<Arthmetic>
    {
        public MCvScalar MCvScalar { get; set; }
        public double Value
        {
            get { return MCvScalar.V0; }
            set { MCvScalar = new MCvScalar(value); }
        }

        public int Dimension => 1;

        public Arthmetic(double x = 0)
        {
            MCvScalar = new MCvScalar(x);
        }

        public static implicit operator double(Arthmetic x)
        {
            return x.Value;
        }

        public static implicit operator Arthmetic(double x)
        {
            return new Arthmetic(x);
        }

        public static implicit operator Gray(Arthmetic x)
        {
            return new Gray(x.Value);
        }

        public static implicit operator Arthmetic(Gray x)
        {
            return new Arthmetic(x.Intensity);
        }

        public bool Equals(Arthmetic other)
        {
            return Value == other.Value;
        }
    }

    public static class MatrixExtensions
    {
        public static Image<Arthmetic, double> Multiply(this Image<Arthmetic, double> a, Image<Arthmetic, double> b)
        {
            Image<Arthmetic, double> res = new Image<Arthmetic, double>(b.Cols, a.Rows);

            for(int r = 0; r < a.Rows; ++r)
            {
                for(int c = 0; c < b.Cols; ++c)
                {
                    double x = 0.0;
                    for(int i = 0; i < a.Cols; ++i)
                    {
                        x += a[r, i] * b[i, c];
                    }
                    res[r, c] = x;
                }
            }

            return res;
        }
        
        public static Image<Arthmetic, double> Multiply(this Mat a, Image<Arthmetic, double> b)
        {
            return a.ToImage<Arthmetic, double>().Multiply(b);
        }

        public static Image<Arthmetic, double> Multiply(this Image<Arthmetic, double> a, Mat b)
        {
            return a.Multiply(b.ToImage<Arthmetic, double>());
        }

        public static void CopyFromVector(this Image<Arthmetic, double> m, Image<Arthmetic, double> v, bool rowWise = true)
        {
            bool isColumnVector = v.Cols == 1;

            for (int r = 0; r < m.Rows; ++r)
            {
                for (int c = 0; c < m.Cols; ++c)
                {
                    int element = rowWise ? r * m.Cols + c : c * m.Rows + r;
                    int vr = isColumnVector ? element : 0;
                    int vc = isColumnVector ? 0 : element;

                    m[r, c] = v[vr, vc];
                }
            }
        }

        public static Image<Arthmetic, double> T(this Image<Arthmetic, double> m)
        {
            return m.Mat.T().ToImage<Arthmetic, double>();
        }
    }
}
