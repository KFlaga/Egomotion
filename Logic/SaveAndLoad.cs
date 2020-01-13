using Emgu.CV;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Egomotion
{
    public class SaveAndLoad
    {
        public static void SaveCalibration(Stream stream, Mat camMat, Emgu.CV.Util.VectorOfFloat distCoeffs)
        {
            var P = camMat.ToImage<Arthmetic, double>();

            TextWriter writer = new StreamWriter(stream);
            writer.WriteLine(P[0, 0]);
            writer.WriteLine(P[1, 1]);
            writer.WriteLine(P[0, 2]);
            writer.WriteLine(P[1, 2]);
            for (int i = 0; i < distCoeffs.Size; ++i)
            {
                writer.WriteLine(distCoeffs[i]);
            }
            writer.Close();
        }

        public static void LoadCalibration(Stream stream, out Mat camMat, out Emgu.CV.Util.VectorOfFloat distCoeffs)
        {
            var P = new Image<Arthmetic, double>(3, 3);
            var dist = new float[5];

            TextReader reader = new StreamReader(stream);
            P[0, 0] = double.Parse(reader.ReadLine());
            P[1, 1] = double.Parse(reader.ReadLine());
            P[0, 2] = double.Parse(reader.ReadLine());
            P[1, 2] = double.Parse(reader.ReadLine());
            P[2, 2] = 1.0;
            for (int i = 0; i < 5; ++i)
            {
                dist[i] = (float)double.Parse(reader.ReadLine());
            }
            reader.Close();

            camMat = P.Mat;
            distCoeffs = new Emgu.CV.Util.VectorOfFloat(dist);
        }
    }
}
