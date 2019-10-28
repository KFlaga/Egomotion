using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Egomotion
{
    public static class RotationConverter
    {
        public static Emgu.CV.Image<Arthmetic, double> MatrixToEulerXYZ(Emgu.CV.Image<Arthmetic, double> matrix)
        {
            var euler = new Emgu.CV.Image<Arthmetic, double>(1, 3);
            if (matrix[0, 2] < 1.0 - 1e-9)
            {
                if (matrix[0, 2] > -1.0 + 1e-9)
                {
                    euler[1, 0] = Math.Asin(matrix[0, 2]);
                    euler[0, 0] = Math.Atan2(-matrix[1, 2], matrix[2, 2]);
                    euler[2, 0] = Math.Atan2(-matrix[0, 1], matrix[0, 0]);
                }
                else // r02 = −1
                {
                    euler[1, 0] = -Math.PI * 0.5;
                    euler[0, 0] = Math.Atan2(-matrix[1, 0], matrix[1, 1]);
                    euler[2, 0] = 0.0;
                }
            }
            else // r02 = +1
            {
                euler[1, 0] = -Math.PI * 0.5;
                euler[0, 0] = -Math.Atan2(-matrix[1, 0], matrix[1, 1]);
                euler[2, 0] = 0.0;
            }
            return euler;
        }
    }
}
