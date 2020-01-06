using Emgu.CV;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Egomotion
{
    public class ComputeMatrix
    {
        public static Image<Arthmetic, double> F(VectorOfPointF leftPoints, VectorOfPointF rightPoints)
        {
            if(leftPoints.Size < 8 || rightPoints.Size < 8)
            {
                return null;
            }

            Mat F = CvInvoke.FindFundamentalMat(leftPoints, rightPoints);
            if (F.Rows == 0)
            {
                return null;
            }
            var Fi = F.ToImage<Arthmetic, double>();
            Fi = Fi.Mul(1 / Fi.Norm);
            return Fi;
        }

        public static Image<Arthmetic, double> K(double fx, double fy, double px, double py)
        {
            return new Image<Arthmetic, double>(new double[,,] {
                { {fx}, {0}, {px} } ,
                { {0}, {fy}, {py} } ,
                { {0}, {0 }, {1 } } ,
            });
        }

        public static Image<Arthmetic, double> E(Image<Arthmetic, double> F, Image<Arthmetic, double> K)
        {
            return K.T().Multiply(F).Multiply(K);
        }

        public static Image<Arthmetic, double> CrossProductToVector(Image<Arthmetic, double> tx)
        {
            var t = new Image<Arthmetic, double>(1, 3);
            t[0, 0] = tx[2, 1];
            t[1, 0] = tx[0, 2];
            t[2, 0] = tx[1, 0];
            return t;
        }
    }
}
