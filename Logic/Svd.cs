using Emgu.CV;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Egomotion
{
    public class Svd
    {
        public Image<Arthmetic, double> S { get; private set; }
        public Image<Arthmetic, double> U { get; private set; }
        public Image<Arthmetic, double> VT { get; private set; }

        public Svd(IInputArray A)
        {
            Mat S = new Mat();
            Mat U = new Mat();
            Mat VT = new Mat();

            CvInvoke.SVDecomp(A, S, U, VT, Emgu.CV.CvEnum.SvdFlag.Default);

            this.S = S.ToImage<Arthmetic, double>();
            this.U = U.ToImage<Arthmetic, double>();
            this.VT = VT.ToImage<Arthmetic, double>();
        }
    }
}
