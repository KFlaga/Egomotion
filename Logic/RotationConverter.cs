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
        public static Emgu.CV.Image<Arthmetic, double> EulerXYZToMatrix(Emgu.CV.Image<Arthmetic, double> euler)
        {
            var matrix = new Emgu.CV.Image<Arthmetic, double>(3, 3);
            double sx = Math.Sin(euler[0, 0]);
            double cx = Math.Cos(euler[0, 0]);
            double sy = Math.Sin(euler[1, 0]);
            double cy = Math.Cos(euler[1, 0]);
            double sz = Math.Sin(euler[2, 0]);
            double cz = Math.Cos(euler[2, 0]);
            matrix[0, 0] = cy * cz;
            matrix[0, 1] = - cy * sz;
            matrix[0, 2] = sy;
            matrix[1, 0] = cz * sx * sy + cx * sz;
            matrix[1, 1] = cx * cz - sx * sy * sz;
            matrix[1, 2] = - cy * sx;
            matrix[2, 0] = - cx * cz * sy + sz * sz;
            matrix[2, 1] = cz * sx + cx * sy * sz;
            matrix[2, 2] = cx * cy;
            return matrix;
        }

        public static Emgu.CV.Image<Arthmetic, double> MatrixToVector(Emgu.CV.Image<Arthmetic, double> matrix)
        {
            var vector = new Emgu.CV.Image<Arthmetic, double>(1, 3);
            Emgu.CV.RotationVector3D vector3D = new Emgu.CV.RotationVector3D();
            vector3D.RotationMatrix = matrix.Mat;
            return new Emgu.CV.Image<Arthmetic, double>(new double[,,]
            {
                {{vector3D[0, 0]}}, {{vector3D[1, 0]}}, {{vector3D[2, 0]}}
            });
        }
    }
}
