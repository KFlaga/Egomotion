using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Drawing;
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

        public Emgu.CV.Image<Arthmetic, double> TransformationMatrix { get; set; }

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
            DecomposeTransformationMatrices(dataset, interval);
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

                    frame.TransformationMatrix = new Emgu.CV.Image<Arthmetic, double>(4, 4);
                    for (int row = 0; row < 4; ++row)
                    {
                        string line = reader.ReadLine();
                        string[] cols = line.Split();
                        for(int col = 0; col < 4; ++col)
                        {
                            frame.TransformationMatrix[row, col] = double.Parse(cols[col], System.Globalization.CultureInfo.InvariantCulture);
                        }
                    }
                }
            }
        }

        private static void DecomposeTransformationMatrices(Dataset dataset, TimeSpan interval)
        {
            DatasetFrame prev = null;
            foreach (var frame in dataset.Frames)
            {
                var translation = new Emgu.CV.Image<Arthmetic, double>(1, 3);
                for(int i = 0; i < 3; ++i)
                {
                    translation[i, 0] = frame.TransformationMatrix[i, 3];
                }

                var rotationMatrix = frame.TransformationMatrix.GetSubRect(new Rectangle(0, 0, 3, 3));
                var euler = RotationConverter.MatrixToEulerXYZ(rotationMatrix);
                
                OdometerFrame odometry = new OdometerFrame()
                {
                    TimeDiff = interval,
                    Translation = translation,
                    Rotation = euler,
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
                    odometry.TranslationDiff = new Emgu.CV.Image<Arthmetic, double>(1, 3);
                    odometry.RotationDiff = new Emgu.CV.Image<Arthmetic, double>(1, 3);
                }

                frame.Odometry = odometry;
                prev = frame;
            }
        }
    }
}
