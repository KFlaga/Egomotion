using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Optimization;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Egomotion
{
    public class FindTransformation
    {
        public static OdometerFrame GetOdometerFrame(Mat left, Mat right, Emgu.CV.Features2D.Feature2D detector, Image<Arthmetic, double> K, double takeBest = 1.0)
        {
            var match = MatchImagePair.Match(left, right, detector);

            var lps = match.LeftPoints.ToArray().Take((int)(match.LeftPoints.Size * takeBest)).ToArray();
            var rps = match.RightPoints.ToArray().Take((int)(match.RightPoints.Size * takeBest)).ToArray();

            var F = ComputeMatrix.F(new VectorOfPointF(lps), new VectorOfPointF(rps));
            if (F == null)
            {
                return null;
            }

            var E = ComputeMatrix.E(F, K);
            DecomposeToRT(E, out Image<Arthmetic, double> R, out Image<Arthmetic, double> t);
            t = ComputeMatrix.CrossProductToVector(t);

            OdometerFrame odometerFrame = new OdometerFrame();
            odometerFrame.Rotation = RotationConverter.MatrixToEulerXYZ(R);
            odometerFrame.RotationMatrix = R;
            odometerFrame.MatK = K;

            Image<Arthmetic, double> C = ComputeCameraCenter(R, t, K, match);
            odometerFrame.Translation = R.Multiply(C);
            //   odometerFrame.Translation = R.T().Multiply(odometerFrame.Translation);

            return odometerFrame;
        }

        public static void DecomposeToRT(Image<Arthmetic, double> F, out Image<Arthmetic, double> R, out Image<Arthmetic, double> t)
        {
            var svd = new Svd(F);

            Image<Arthmetic, double> W = new Image<Arthmetic, double>(new double[,,] {
                { {0}, {-1 }, {0 } } ,
                { {1}, {0 }, {0 } } ,
                { {0}, {0 }, {1 } } ,
            });

            var R1 = svd.U.Multiply(W.T()).Multiply(svd.VT);
            var R2 = svd.U.Multiply(W).Multiply(svd.VT);

            var I = new Image<Arthmetic, double>(3, 3);
            I[0, 0] = 1;
            I[1, 1] = 1;
            I[2, 2] = 1;

            var s1 = (R1 - I).Norm;
            var s2 = (R2 - I).Norm;
            R = s1 < s2 ? R1 : R2;
            
            double ss = (svd.S[0, 0] + svd.S[1, 0]) / 2;

            Image<Arthmetic, double> Z = new Image<Arthmetic, double>(new double[,,] {
                { {0}, {-ss }, {0 } } ,
                { {ss}, {0 }, {0 } } ,
                { {0}, {0 }, {0 } } ,
            });

            t = svd.U.Multiply(Z).Multiply(svd.U.T());
        }

        public static Image<Arthmetic, double> ComputeCameraCenter(Image<Arthmetic, double> R, Image<Arthmetic, double> T, Image<Arthmetic, double> K, MacthingResult match, double takeBest = 0.25)
        {
            Image<Arthmetic, double> Kinv = new Image<Arthmetic, double>(3, 3);
            CvInvoke.Invert(K, Kinv, Emgu.CV.CvEnum.DecompMethod.LU);

            Image<Arthmetic, double> C = R.T().Multiply(T);

            if(Math.Abs(C[2, 0]) < 1e-8)
            {
                // TODO: alternative for such case
                throw new NotImplementedException("Initial camera center has zero Z");
            }
            
            double alfa = C[0, 0] / C[2, 0];
            double beta = C[1, 0] / C[2, 0];
            
            var lps = match.LeftPoints.ToArray().Take((int)(match.LeftPoints.Size * takeBest)).ToArray();
            var rps = match.RightPoints.ToArray().Take((int)(match.RightPoints.Size * takeBest)).ToArray();

            double[] czs = new double[lps.Length];

            for (int i = 0; i < lps.Length; ++i)
            {
                Image<Arthmetic, double> p1 = match.LeftPoints[i].ToVector();
                Image<Arthmetic, double> p2 = match.RightPoints[i].ToVector();

                Image<Arthmetic, double> P1 = Kinv.Multiply(p1);
                Image<Arthmetic, double> P2 = Kinv.Multiply(p2);

                double czX = (R[0, 0] * P1[0, 0] + R[0, 1] * P1[1, 0] + R[0, 2] * P1[2, 0] - P2[0, 0]) / 
                                          (R[0, 0] * alfa + R[0, 1] * beta + R[0, 2]);
                double czY = (R[1, 0] * P1[0, 0] + R[1, 1] * P1[1, 0] + R[1, 2] * P1[2, 0] - P2[1, 0]) /
                                          (R[1, 0] * alfa + R[1, 1] * beta + R[1, 2]);
                double czZ = (R[2, 0] * P1[0, 0] + R[2, 1] * P1[1, 0] + R[2, 2] * P1[2, 0] - P2[2, 0]) /
                                          (R[2, 0] * alfa + R[2, 1] * beta + R[2, 2]);

                czs[i] = (czX + czY + czZ) / 3;
            }

            double cz = czs.Aggregate((a, b) => a + b) / czs.Length;
            
            C[0, 0] = cz * alfa;
            C[1, 0] = cz * beta;
            C[2, 0] = cz;

            return C;
        }
    }
 }
