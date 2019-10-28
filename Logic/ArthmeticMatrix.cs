using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
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
}
