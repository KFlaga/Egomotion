using Microsoft.Win32;
using System;
using System.IO;
using System.Windows;
using Microsoft.WindowsAPICodePack.Dialogs;

namespace Egomotion
{
    public static class FileOp
    {
        public delegate void FileDelegate(Stream file, string filePath);

        public const string AllFilesFilter = "All files|*.*";

        public static void LoadFromFile(FileDelegate onFileOpen, string filter = AllFilesFilter)
        {
            OpenFileDialog fileDialog = new OpenFileDialog();
            fileDialog.Filter = filter;
            bool? res = fileDialog.ShowDialog();
            if(res != null && res == true && File.Exists(fileDialog.FileName))
            {
                Stream fs = fileDialog.OpenFile();
                try
                {
                    onFileOpen(fs, fileDialog.FileName);
                }
                catch(Exception exc)
                {
                    MessageBox.Show("Error while reading from file: " + exc.Message, "Error");
                }
                fs.Close();
            }
        }

        public static void SaveToFile(FileDelegate onFileOpen, string filter = AllFilesFilter)
        {
            SaveFileDialog fileDialog = new SaveFileDialog();
            fileDialog.Filter = filter;
            bool? res = fileDialog.ShowDialog();
            if(res != null && res == true)
            {
                Stream fs = fileDialog.OpenFile();
                try
                {
                    onFileOpen(fs, fileDialog.FileName);
                }
                catch(Exception exc)
                {
                    MessageBox.Show("Error while writing to file: " + exc.Message, "Error");
                }
                fs.Close();
            }
        }

        public static void OpenFolder(Action<string> onFolderOpen)
        {
            CommonOpenFileDialog fileDialog = new CommonOpenFileDialog();
            fileDialog.IsFolderPicker = true;
            fileDialog.EnsurePathExists = true;
            var res = fileDialog.ShowDialog();
            if (res == CommonFileDialogResult.Ok)
            {
                try
                {
                    onFolderOpen(fileDialog.FileName);
                }
                catch (Exception exc)
                {
                    MessageBox.Show("Error while reading from file: " + exc.Message, "Error");
                }
            }
        }
    }
}
