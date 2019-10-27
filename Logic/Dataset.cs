using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace Egomotion
{
    public class DatasetFrame
    {
        public string ImageFile { get; set; }
        public string CameraFile { get; set; }

        public Emgu.CV.Image<Gray, double> ProjectionMatrix { get; set; }

        public OdometerFrame Odometry { get; set; }
    }

    public class Dataset
    {
        public List<DatasetFrame> Frames { get; set; }

        public static Dataset Load(string rootDir, TimeSpan interval)
        {
            var subdirs = Directory.EnumerateDirectories(rootDir).Select((x) => Path.GetFileName(x));
            if (!subdirs.Contains("images") || !subdirs.Contains("cameras"))
            {
                throw new InvalidDataException("Dataset is missing 'images' or 'cameras' subfolder.");
            }
            
            Dictionary<int, DatasetFrame> frames = LoadDatasetFrames(rootDir);
            if(frames.Count < 2)
            {
                throw new InvalidDataException("Dataset needs at least two frames.");
            }

            Dataset dataset = new Dataset() { Frames = new List<DatasetFrame>(frames.Count) };
            for (int i = 0; i < frames.Count; ++i)
            {
                dataset.Frames.Add(frames[i]);
            }

            LoadProjectionMatrices(dataset);
            DecomposeProjectionMatrices(dataset, interval);
            return dataset;
        }

        private static int FrameNumber(string fileName)
        {
            return int.Parse(Path.GetFileNameWithoutExtension(fileName));
        }

        private static Dictionary<int, DatasetFrame> LoadDatasetFrames(string rootDir)
        {
            string imgDir = Path.Combine(rootDir, "images");
            string camDir = Path.Combine(rootDir, "cameras");

            var images = Directory.EnumerateFiles(imgDir);

            Dictionary<int, DatasetFrame> frames = new Dictionary<int, DatasetFrame>();
            foreach (string imgFile in images)
            {
                int number = FrameNumber(imgFile);
                frames[number] = new DatasetFrame() { ImageFile = Path.Combine(imgDir, imgFile), CameraFile = Path.Combine(camDir, number.ToString() + ".txt") };
            }
            return frames;
        }

        private static void LoadProjectionMatrices(Dataset dataset)
        {
            foreach(var frame in dataset.Frames)
            {
                using (Stream s = new FileStream(frame.CameraFile, FileMode.Open))
                {
                    TextReader reader = new StreamReader(s);

                    frame.ProjectionMatrix = new Emgu.CV.Image<Gray, double>(4, 4);
                    for (int row = 0; row < 4; ++row)
                    {
                        string line = reader.ReadLine();
                        string[] cols = line.Split();
                        for(int col = 0; col < 4; ++col)
                        {
                            frame.ProjectionMatrix[row, col] = new Gray(double.Parse(cols[col], System.Globalization.CultureInfo.InvariantCulture));
                        }
                    }
                }
            }
        }

        private static void DecomposeProjectionMatrices(Dataset dataset, TimeSpan interval)
        {
            DatasetFrame prev = null;
            foreach (var frame in dataset.Frames)
            {
                // Emgu.CV doesn't support decomposeProjectionMatrix function, so 2nd opencv wrapper is used here as well
                // We may decide to switch to OpenCvSharp only if it offers all required functions

                double[,] projectionMatrix = new double[3, 4]; // Last row is skipped
                for(int r = 0; r < 3; ++r)
                {
                    for(int c = 0; c < 4; ++c)
                    {
                        projectionMatrix[r, c] = frame.ProjectionMatrix[r, c].Intensity;
                    }
                }

                OpenCvSharp.Cv2.DecomposeProjectionMatrix(projectionMatrix,
                    out double[,] camera,
                    out double[,] rotation,
                    out double[] translation,
                    out double[,] rx,
                    out double[,] ry,
                    out double[,] rz,
                    out double[] eulerAngles
                );

                var emguTranslation = new Emgu.CV.Image<Gray, double>(3, 1);
                var emguRotation = new Emgu.CV.Image<Gray, double>(3, 1);
                for (int i = 0; i < 3; ++i)
                {
                    emguTranslation[0, i] = new Gray(translation[i] / translation[3]);
                    emguRotation[0, i] = new Gray(eulerAngles[i]);
                }

                OdometerFrame odometry = new OdometerFrame()
                {
                    TimeDiff = interval,
                    Translation = emguTranslation,
                    Rotation = emguRotation,
                    Velocity = null,
                    AngularVelocity = null,
                };

                if(prev != null)
                {
                    odometry.TranslationDiff = odometry.Translation - prev.Odometry.Translation;
                    odometry.RotationDiff = odometry.Rotation - prev.Odometry.Rotation;
                }
                else
                {
                    odometry.TranslationDiff = new Emgu.CV.Image<Gray, double>(1, 3);
                    odometry.RotationDiff = new Emgu.CV.Image<Gray, double>(1, 3);
                }

                frame.Odometry = odometry;
                prev = frame;
            }
        }
    }
}
