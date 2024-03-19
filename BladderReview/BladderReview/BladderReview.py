import logging
import os

import vtk
import pathlib
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import ctk
import qt
import SegmentStatistics


try:
    import pandas as pd
    import numpy as np
    import SimpleITK as sitk
    from datetime import datetime
except:
    slicer.util.pip_install('pandas')
    slicer.util.pip_install('numpy')
    slicer.util.pip_install('SimpleITK')
    slicer.util.pip_install('datetime')

    import pandas as pd
    import numpy as np
    import SimpleITK as sitk
    from datetime import datetime


#
# BladderReview
#

class BladderReview(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "BladderReview"
        self.parent.categories = ["Examples"]
        self.parent.dependencies = []
        self.parent.contributors = ["Anna Zapaishchykova (BWH), Dr. Benjamin H. Kann, AIM-Harvard"]
        self.parent.helpText = """
Slicer3D extension for rating using Likert-type score Deep-learning generated segmentations, with segment editor funtionality. 
Created to speed up the validation process done by a clinician - the dataset loads in one batch with no need to load masks and volumes separately.
It is important that each nii file has a corresponding mask file with the same name and the suffix _mask.nii
"""

        self.parent.acknowledgementText = """
This file was developed by Anna Zapaishchykova, BWH. 
"""


#
# BladderReviewWidget
#

class BladderReviewWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False
        self.volume_node = None
        self.volume_node_sag = None
        self.volume_node_cor = None
        self.segmentation_node = None
        self.segmentation_visible = False
        self.segmentation_color = [1, 0, 0]
        self.nifti_files = []
        self.nifti_files_sag = []
        self.nifti_files_cor = []
        self.segmentation_files = []
        self.directory = None
        self.current_index = 0
        self.intensities = []
        self.n_files = 0
        self.current_df = None
        self.segmentEditorNode = None
        self.red_widget = None
        self.yellow_widget = None
        self.green_widget = None

        # Set own parameters
        self.likert_scores_1 = []       # Evident tumor on T2W
        self.likert_scores_2 = []       # Evident tumor on DWI
        self.likert_scores_3 = []       # Evident tumor on DCE
        self.likert_scores_5 = []       # Residual Tumor
        self.likert_scores_6 = []       # Involvement of Muscularis propria
        self.likert_scores_7 = []       # Radiological downstaging
        self.likert_scores_8 = []       # Response assessment
        self.likert_scores_9 = []       # Likert Segmentation
        self.staging_T = []
        self.staging_N = []
        self.VIRADS = []
        self.nacVIRADS = []

        self.start_time = None
        self.end_time = None
        self.diameter_measurement_complete = False
        self.diameter_measurement_2_complete = False
        self.diameter_measurement_3_complete = False
        self.coordinates_list_length = []
        self.coordinates_list_width = []
        self.coordinates_list_height = []
        self.lineNode_length = None                                             # Variable to store the first line node (for diameter measurement)
        self.lineNode_width = None                                              # Variable to store the second line node (for diameter measurement)
        self.lineNode_height = None

        self.start_time_load_volume = None                                      # Timing for when volume is loaded
        self.end_time_load_volume = None                                        # Timing for when BladderReview is finished
        self.start_time_segmentation = None
        self.end_time_segmentation = None

        self.dummy_radio_buttons = []

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        import qSlicerSegmentationsModuleWidgetsPythonQt
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/BladderReview.ui'))

        # Layout within the collapsible button
        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Input path"
        self.layout.addWidget(parametersCollapsibleButton)

        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

        self.atlasDirectoryButton = ctk.ctkDirectoryButton()
        parametersFormLayout.addRow("Directory: ", self.atlasDirectoryButton)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = SlicerLikertDLratingLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        self.ui.PathLineEdit = ctk.ctkDirectoryButton()

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.atlasDirectoryButton.directoryChanged.connect(self.onAtlasDirectoryChanged)
        self.ui.save_and_next.connect('clicked(bool)', self.save_and_next_clicked)
        #self.ui.overwrite_mask.connect('clicked(bool)', self.overwrite_mask_clicked)

        # Create a new segment editor widget and add it to the NiftyViewerWidget
        self._createSegmentEditorWidget_()

        # self.editorWidget.volumes.collapsed = True
        # Set parameter node first so that the automatic selections made when the scene is set are saved

        # Make sure parameter node is initialized (needed for module reload)
        # self.initializeParameterNode()

        # Set up diameter measurement button
        self.ui.diameter_length.clicked.connect(self.start_diameter_measurement)
        self.ui.diameter_width.clicked.connect(self.start_diameter_measurement_2)
        self.ui.diameter_height.clicked.connect(self.start_diameter_measurement_3)

        # Set up quality assessment complete button
        self.ui.completeButton.clicked.connect(self.complete_quality_assessment)

        self.dummy_radio_buttons = [
            self.ui.radioButton_0_dummy,
            self.ui.radioButton_1_dummy,
            self.ui.radioButton_2_dummy,
            self.ui.radioButton_3_dummy,
            self.ui.radioButton_5_dummy,
            self.ui.radioButton_6_dummy,
            self.ui.radioButton_7_dummy,
            self.ui.radioButton_8_dummy,
            self.ui.radioButton_9_dummy,
            self.ui.radioButton_11_dummy,
            self.ui.radioButton_12_dummy,
            self.ui.radioButton_13_dummy
        ]
        # Save and Next
        # Overwrite Edited Mask

    def _createSegmentEditorWidget_(self):
        """Create and initialize a customize Slicer Editor which contains just some the tools that we need for the segmentation"""

        import qSlicerSegmentationsModuleWidgetsPythonQt

        # advancedCollapsibleButton
        self.segmentEditorWidget = qSlicerSegmentationsModuleWidgetsPythonQt.qMRMLSegmentEditorWidget(
        )
        self.segmentEditorWidget.setMaximumNumberOfUndoStates(10)
        self.segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        self.segmentEditorWidget.unorderedEffectsVisible = False
        self.segmentEditorWidget.setEffectNameOrder([
            'Paint', 'Draw', 'Erase',
        ])
        self.layout.addWidget(self.segmentEditorWidget)

    def overwrite_mask_clicked(self):
        # overwrite self.segmentEditorWidget.segmentationNode()
        print("Saved segmentation",
              self.segmentation_files[self.current_index].split("/")[-1].split(".")[0] + "_upd.nii.gz")
        # segmentation_node = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLSegmentationNode')

        # Get the file path where you want to save the segmentation node
        file_path = self.directory + "/t.seg.nrrd"

        i = 1
        file_path_nifti = self.directory + "/" + self.segmentation_files[self.current_index].split("/")[-1].split(".nii.gz")[0] + "_v" + str(i) + ".nii.gz"

        # # Save the segmentation node to file as nifti
        # file_path_nifti = self.directory + "/" + self.segmentation_files[self.current_index].split("/")[-1].split(".")[
        #     0] + "_upd.nii.gz"
        # Save the segmentation node to file
        slicer.util.saveNode(self.segmentation_node, file_path)

        img = sitk.ReadImage(file_path)

        while os.path.exists(file_path_nifti):
            i += 1
            file_path_nifti = self.directory + "/" + self.segmentation_files[self.current_index].split("/")[-1].split(".nii.gz")[0] + "_v" + str(i) + ".nii.gz"

        print('Saving segmentation to file: ', file_path_nifti)
        sitk.WriteImage(img, file_path_nifti)

    def onAtlasDirectoryChanged(self, directory):
        if self.volume_node:
            slicer.mrmlScene.RemoveNode(self.volume_node)
        if self.segmentation_node:
            slicer.mrmlScene.RemoveNode(self.segmentation_node)
        if self.volume_node_sag:
            slicer.mrmlScene.RemoveNode(self.volume_node_sag)
        if self.volume_node_cor:
            slicer.mrmlScene.RemoveNode(self.volume_node_cor)

        self.directory = directory

        # load the .csv file with the old annotations or create a new one
        if os.path.exists(directory + "/annotations.csv"):
            self.current_index = pd.read_csv(directory + "/annotations.csv").shape[0]
            print("Restored current index: ", self.current_index)
        else:
            self.current_df = pd.DataFrame(columns=['file', 'annotation'])
            self.current_index = 0

        # count the number of files in the directory
        file_pairs = [file for file in os.listdir(directory) if "_tT2" in file or "_sT2" in file or '_cT2' in file]
        file_pairs.sort()  # Ensure T1 and T2 for each patient are next to each other
        print(file_pairs)

        for i in range(0, len(file_pairs)):
            if "_tT2" in file_pairs[i]:
                self.nifti_files.append(directory + "/" + file_pairs[i])
                self.segmentation_files.append((directory + "/" + file_pairs[i]).replace("tT2", "mask"))
            else:
                print("Unexpected file: ", file_pairs[i])

            if "_sT2" in file_pairs[i]:
                self.nifti_files_sag.append(directory + "/" + file_pairs[i])
            else:
                print("No Sagital file for: ", file_pairs[i])

            if "_cT2" in file_pairs[i]:
                self.nifti_files_cor.append(directory + "/" + file_pairs[i])
            else:
                print("No Coronal file for: ", file_pairs[i])

        self.n_files = len(self.nifti_files)  # Now n_files should be equal to the number of T1 (or T2) files
        self.ui.status_checked.setText("Checking: " + str(self.current_index + 1) + " / " + str(self.n_files))

        # load first the volume node
        self.load_nifti_file()

    def load_nifti_file(self):
        # Timing
        current_time = datetime.now()
        self.start_time_load_volume = current_time.strftime('%Y-%m-%d %H:%M:%S')

        # Reset the slice views to clear any remaining segmentations
        slicer.util.resetSliceViews()

        # Axial scan
        file_path_ax = self.nifti_files[self.current_index]
        if self.volume_node:
            slicer.mrmlScene.RemoveNode(self.volume_node)
        if self.segmentation_node:
            slicer.mrmlScene.RemoveNode(self.segmentation_node)
        if self.segmentEditorNode:
            slicer.mrmlScene.RemoveNode(self.segmentEditorNode)

        self.volume_node = slicer.util.loadVolume(file_path_ax)
        slicer.app.applicationLogic().PropagateVolumeSelection(0)

        # Sagital scan
        file_path_sag = self.nifti_files_sag[self.current_index]
        if self.volume_node_sag:
            slicer.mrmlScene.RemoveNode(self.volume_node_sag)

        self.volume_node_sag = slicer.util.loadVolume(file_path_sag)
        slicer.app.applicationLogic().PropagateVolumeSelection(0)

        # Coronal scan
        file_path_cor = self.nifti_files_cor[self.current_index]
        if self.volume_node_cor:
            slicer.mrmlScene.RemoveNode(self.volume_node_cor)

        self.volume_node_cor = slicer.util.loadVolume(file_path_cor)
        slicer.app.applicationLogic().PropagateVolumeSelection(0)

        # Set the layout to show 4 views stacked on top of each other
        lm = slicer.app.layoutManager()
        lm.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)


        # set the sagital volume to the yellow slice view
        self.yellow_widget = lm.sliceWidget('Yellow')
        yellow_logic = self.yellow_widget.sliceLogic()
        self.yellow_slice_node = yellow_logic.GetSliceNode()
        #self.yellow_slice_node.SetOrientationToSagittal()
        yellow_cn = yellow_logic.GetSliceCompositeNode()
        yellow_cn.SetBackgroundVolumeID(self.volume_node_sag.GetID())

        # set the coronal volume to the green slice view
        self.green_widget = lm.sliceWidget('Green')
        green_logic = self.green_widget.sliceLogic()
        self.green_slice_node = green_logic.GetSliceNode()
        #self.green_slice_node.SetOrientationToCoronal()
        green_cn = green_logic.GetSliceCompositeNode()
        green_cn.SetBackgroundVolumeID(self.volume_node_cor.GetID())

        # set the axial volume to the red slice view
        self.red_widget = lm.sliceWidget('Red')
        red_logic = self.red_widget.sliceLogic()
        self.red_slice_node = red_logic.GetSliceNode()
        #self.red_slice_node.SetOrientationToAxial()
        red_cn = red_logic.GetSliceCompositeNode()
        red_cn.SetBackgroundVolumeID(self.volume_node.GetID())

    def save_and_next_clicked(self):
        current_time = datetime.now()
        self.end_time_segmentation = current_time.strftime('%Y-%m-%d %H:%M:%S')

        self.overwrite_mask_clicked()

        # Save Likert score visibility DCE
        likert_score_9 = 0
        if self.ui.radioButton_9_1.isChecked():
            likert_score_9 = 1
        elif self.ui.radioButton_9_2.isChecked():
            likert_score_9 = 2
        elif self.ui.radioButton_9_3.isChecked():
            likert_score_9 = 3
        elif self.ui.radioButton_9_4.isChecked():
            likert_score_9 = 4
        elif self.ui.radioButton_9_5.isChecked():
            likert_score_9 = 5

        self.likert_scores_9 = likert_score_9

        # append data frame to CSV file
        data = {'file': [self.nifti_files[self.current_index].split("/")[-1]],
                'Staging cTx': [self.staging_T],
                'Staging cNx': [self.staging_N],
                'Visibility T2W': [self.likert_scores_1],
                'Visibility DWI': [self.likert_scores_2],
                'Visibility DCE': [self.likert_scores_3],
                'diameter_length': [self.ui.measured_diameter1.toPlainText()],
                'diameter_width': [self.ui.measured_diameter2.toPlainText()],
                'diameter_height': [self.ui.measured_diameter3.toPlainText()],
                'coordinates length': [self.coordinates_list_length],
                'coordinates width': [self.coordinates_list_width],
                'coordinates height': [self.coordinates_list_height],
                'comments_diameter': [self.ui.comment_diameter.toPlainText()],
                'VIRADS': [self.VIRADS],
                'nacVIRADS': [self.nacVIRADS],
                'Residual tumor': [self.likert_scores_5],
                'Involvement Muscularis Propria': [self.likert_scores_6],
                'Radiological Downstaging': [self.likert_scores_7],
                'Response assessment': [self.likert_scores_8],
                'start_time_quality_assessment': [self.start_time_load_volume],
                'end_time_quality_assessment': [self.end_time_load_volume],
                'Likert Segmentation': [self.likert_scores_9],
                'comment': [self.ui.comment.toPlainText()],
                'start_time_segmentation': [self.start_time_segmentation],
                'end_time_segmentation': [self.end_time_segmentation]}

        df = pd.DataFrame(data)
        file_exists = os.path.isfile(self.directory + "/annotations.csv")
        df.to_csv(self.directory + "/annotations.csv", mode='a', index=False, header=not file_exists)

        self.ui.groupbox0.setEnabled(True)  # Enable the entire QGroupBox
        self.ui.groupbox1.setEnabled(True)  # Enable the entire QGroupBox
        self.ui.groupbox2.setEnabled(True)  # Enable the entire QGroupBox
        self.ui.groupbox3.setEnabled(True)  # Enable the entire QGroupBox
        self.ui.groupbox4.setEnabled(True)  # Enable the entire QGroupBox
        self.ui.groupbox5.setEnabled(True)  # Enable the entire QGroupBox
        self.ui.groupbox6.setEnabled(True)  # Enable the entire QGroupBox
        self.ui.groupbox7.setEnabled(True)  # Enable the entire QGroupBox
        self.ui.groupbox8.setEnabled(True)  # Enable the entire QGroupBox
        self.ui.groupbox11.setEnabled(True)
        self.ui.groupbox12.setEnabled(True)
        self.ui.groupbox13.setEnabled(True)


        slicer.mrmlScene.RemoveNode(self.segmentation_node)
        self.resetUIElements()

        # go to the next file if there is one
        if self.current_index < self.n_files - 1:
            self.current_index += 1
            self.load_nifti_file()
            self.ui.status_checked.setText("Checking: " + str(self.current_index + 1) + " / " + str(self.n_files))
            self.ui.comment.setPlainText("")
        else:
            slicer.util.messageBox("All files checked, continue with next patient")

    def start_diameter_measurement(self):
        self.diameter_measurement_complete = False  # Add this line
        # Remove existing lineNode if it exists
        if hasattr(self, 'lineNode_length') and self.lineNode_length:
            slicer.mrmlScene.RemoveNode(self.lineNode_length)
            self.lineNode_length = None

        self.lineNode_length = slicer.mrmlScene.CreateNodeByClass("vtkMRMLMarkupsLineNode")
        self.lineNode_length.SetName("temp")
        slicer.mrmlScene.AddNode(self.lineNode_length)

        # Create a display node for the line node
        displayNode = self.lineNode_length.GetMarkupsDisplayNode()
        if not displayNode:
            displayNode = self.lineNode_length.CreateDefaultDisplayNodes()

        # Set the desired point size
        displayNode.SetGlyphScale(1.0)  # Adjust the scale as needed
        displayNode.SetTextScale(0)

        self.lineNode_length.SetName(slicer.mrmlScene.GetUniqueNameByString("diameter"))

        # setup placement
        slicer.modules.markups.logic().SetActiveListID(self.lineNode_length)
        interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
        interactionNode.SwitchToSinglePlaceMode()

        # Define a callback function that calculates and displays the line length
        def onPointAdded(vtkObject, event):
            # Ensure the line has at least 2 points before trying to calculate length
            if self.lineNode_length.GetNumberOfDefinedControlPoints() >= 2:
                diameter = round(self.lineNode_length.GetLineLengthWorld(), 4)
                self.ui.measured_diameter1.setPlainText(str(diameter))

                # Collect and store the coordinates of the control points
                coordinates_length = []
                for i in range(self.lineNode_length.GetNumberOfControlPoints()):
                    controlPoint = [0, 0, 0]
                    self.lineNode_length.GetNthControlPointPositionWorld(i, controlPoint)
                    coordinates_length.append(controlPoint)
                self.coordinates_list_length.append(coordinates_length)

        # Add an observer that triggers the callback function every time a point is added to the line
        self.lineNode_length.AddObserver(self.lineNode_length.PointModifiedEvent, onPointAdded)

    def start_diameter_measurement_2(self):
        self.diameter_measurement_2_complete = False
        if hasattr(self, 'lineNode_width') and self.lineNode_width:
            self.lineNode_length.SetLocked(True)
            self.lineNode_width = None

        self.lineNode_width = slicer.mrmlScene.CreateNodeByClass("vtkMRMLMarkupsLineNode")
        self.lineNode_width.SetName("temp")
        slicer.mrmlScene.AddNode(self.lineNode_width)

        # Create a display node for the line node
        displayNode = self.lineNode_width.GetMarkupsDisplayNode()
        if not displayNode:
            displayNode = self.lineNode_width.CreateDefaultDisplayNodes()

        # Set the desired point size
        displayNode.SetGlyphScale(1.0)  # Adjust the scale as needed
        displayNode.SetTextScale(0)

        self.lineNode_width.SetName(slicer.mrmlScene.GetUniqueNameByString("diameter"))


        # setup placement
        slicer.modules.markups.logic().SetActiveListID(self.lineNode_width)
        interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
        interactionNode.SwitchToSinglePlaceMode()

        # Define a callback function that calculates and displays the line length
        def onPointAdded(vtkObject, event):
            # Ensure the line has at least 2 points before trying to calculate length
            if self.lineNode_width.GetNumberOfDefinedControlPoints() >= 2:
                diameter = round(self.lineNode_width.GetLineLengthWorld(), 4)
                self.ui.measured_diameter2.setPlainText(str(diameter))

                # Collect and store the coordinates of the control points
                coordinates_width = []
                for i in range(self.lineNode_width.GetNumberOfControlPoints()):
                    controlPoint = [0, 0, 0]
                    self.lineNode_width.GetNthControlPointPositionWorld(i, controlPoint)
                    coordinates_width.append(controlPoint)
                self.coordinates_list_width.append(coordinates_width)

        # Add an observer that triggers the callback function every time a point is added to the line
        self.lineNode_width.AddObserver(self.lineNode_width.PointModifiedEvent, onPointAdded)

    def start_diameter_measurement_3(self):
        self.diameter_measurement_3_complete = False
        if hasattr(self, 'lineNode_height') and self.lineNode_height:
            self.lineNode_length.SetLocked(True)
            self.lineNode_width.SetLocked(True)
            self.lineNode_height = None

        self.lineNode_height = slicer.mrmlScene.CreateNodeByClass("vtkMRMLMarkupsLineNode")
        self.lineNode_height.SetName("temp")
        slicer.mrmlScene.AddNode(self.lineNode_height)

        # Create a display node for the line node
        displayNode = self.lineNode_height.GetMarkupsDisplayNode()
        if not displayNode:
            displayNode = self.lineNode_height.CreateDefaultDisplayNodes()

        # Set the desired point size
        displayNode.SetGlyphScale(1.0)  # Adjust the scale as needed
        displayNode.SetTextScale(0)

        self.lineNode_height.SetName(slicer.mrmlScene.GetUniqueNameByString("diameter"))


        # setup placement
        slicer.modules.markups.logic().SetActiveListID(self.lineNode_height)
        interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
        interactionNode.SwitchToSinglePlaceMode()

        # Define a callback function that calculates and displays the line length
        def onPointAdded(vtkObject, event):
            # Ensure the line has at least 2 points before trying to calculate length
            if self.lineNode_height.GetNumberOfDefinedControlPoints() >= 2:
                diameter = round(self.lineNode_height.GetLineLengthWorld(), 4)
                self.ui.measured_diameter3.setPlainText(str(diameter))

                # Collect and store the coordinates of the control points
                coordinates_height = []
                for i in range(self.lineNode_height.GetNumberOfControlPoints()):
                    controlPoint = [0, 0, 0]
                    self.lineNode_height.GetNthControlPointPositionWorld(i, controlPoint)
                    coordinates_height.append(controlPoint)
                self.coordinates_list_height.append(coordinates_height)

        # Add an observer that triggers the callback function every time a point is added to the line
        self.lineNode_height.AddObserver(self.lineNode_height.PointModifiedEvent, onPointAdded)

    def complete_quality_assessment(self):
        # Save lineNodes
        object_path_length = self.directory + "/" + self.segmentation_files[self.current_index].split("/")[
            -1].split('_')[1] + "_diameter_objects" + "/object_diameter_length.mrk.json"

        if self.lineNode_length:
            slicer.util.saveNode(self.lineNode_length, object_path_length)
        elif not self.lineNode_length and not self.ui.measured_diameter1.toPlainText():
            slicer.util.errorDisplay("No diameter measurement provided for length, please do so before proceeding")
            return

        object_path_width = self.directory + "/" + self.segmentation_files[self.current_index].split("/")[
            -1].split('_')[1] + "_diameter_objects" + "/object_diameter_width.mrk.json"

        if self.lineNode_width:
            slicer.util.saveNode(self.lineNode_width, object_path_width)
        elif not self.lineNode_width and not self.ui.measured_diameter2.toPlainText():
            slicer.util.errorDisplay("No diameter measurement provided for width, please do so before proceeding")
            return

        object_path_height = self.directory + "/" + self.segmentation_files[self.current_index].split("/")[
            -1].split('_')[1] + "_diameter_objects" + "/object_diameter_height.mrk.json"
        if self.lineNode_height:
            slicer.util.saveNode(self.lineNode_height, object_path_height)
        elif not self.lineNode_height and not self.ui.measured_diameter3.toPlainText():
            slicer.util.errorDisplay("No diameter measurement provided for height, please do so before proceeding")
            return

        current_time = datetime.now()
        self.end_time_load_volume = current_time.strftime('%Y-%m-%d %H:%M:%S')
        self.start_time_segmentation = current_time.strftime('%Y-%m-%d %H:%M:%S')

        self.save_variables()

        # Lock the diameter measurement AND Disable all groupboxes
        self.diameter_measurement_complete = True
        self.diameter_measurement_2_complete = True
        self.diameter_measurement_3_complete = True

        self.ui.groupbox0.setEnabled(False)
        self.ui.groupbox1.setEnabled(False)  # Disable the entire QGroupBox
        self.ui.groupbox2.setEnabled(False)  # Disable the entire QGroupBox
        self.ui.groupbox3.setEnabled(False)  # Disable the entire QGroupBox
        self.ui.groupbox4.setEnabled(False)  # Disable the entire QGroupBox
        self.ui.groupbox5.setEnabled(False)  # Disable the entire QGroupBox
        self.ui.groupbox6.setEnabled(False)  # Disable the entire QGroupBox
        self.ui.groupbox7.setEnabled(False)  # Disable the entire QGroupBox
        self.ui.groupbox8.setEnabled(False)  # Disable the entire QGroupBox
        self.ui.groupbox11.setEnabled(False)
        self.ui.groupbox12.setEnabled(False)
        self.ui.groupbox13.setEnabled(False)


        # Remove existing lineNode if it exists
        if hasattr(self, 'lineNode_length') and self.lineNode_length:
            slicer.mrmlScene.RemoveNode(self.lineNode_length)
            self.lineNode_length = None

        # Remove existing lineNode if it exists
        if hasattr(self, 'lineNode_width') and self.lineNode_width:
            slicer.mrmlScene.RemoveNode(self.lineNode_width)
            self.lineNode_width = None

        # Remove existing lineNode if it exists
        if hasattr(self, 'lineNode_height') and self.lineNode_height:
            slicer.mrmlScene.RemoveNode(self.lineNode_height)
            self.lineNode_height = None

        # After completion, load segmentation mask
        self.load_nifti_segmentation()


    def save_variables(self):
        # Save staging cTx:
        likert_score_0 = 'x'
        if self.ui.radioButton_0_1.isChecked():
            likert_score_0 = 'T0'
        elif self.ui.radioButton_0_2.isChecked():
            likert_score_0 = 'Ta'
        elif self.ui.radioButton_0_3.isChecked():
            likert_score_0 = 'T1'
        elif self.ui.radioButton_0_4.isChecked():
            likert_score_0 = 'T2'
        elif self.ui.radioButton_0_5.isChecked():
            likert_score_0 = 'T3'
        elif self.ui.radioButton_0_6.isChecked():
            likert_score_0 = 'T4a'
        elif self.ui.radioButton_0_7.isChecked():
            likert_score_0 = 'T4b'

        self.staging_T = likert_score_0

        # Save staging cNx:
        likert_score_11 = 'x'
        if self.ui.radioButton_11_1.isChecked():
            likert_score_11 = 'N0'
        elif self.ui.radioButton_11_2.isChecked():
            likert_score_11 = 'N1'
        elif self.ui.radioButton_11_3.isChecked():
            likert_score_11 = 'N2'
        elif self.ui.radioButton_11_4.isChecked():
            likert_score_11 = 'N3'

        self.staging_N = likert_score_11

        # Save Likert score visibility T2W
        likert_score_1 = 0
        if self.ui.radioButton_1_1.isChecked():
            likert_score_1 = 1
        elif self.ui.radioButton_1_2.isChecked():
            likert_score_1 = 2
        elif self.ui.radioButton_1_3.isChecked():
            likert_score_1 = 3
        elif self.ui.radioButton_1_4.isChecked():
            likert_score_1 = 4
        elif self.ui.radioButton_1_5.isChecked():
            likert_score_1 = 5

        self.likert_scores_1 = likert_score_1
        #self.likert_scores_1.append([self.current_index, likert_score_1])

        # Save Likert score visibility DWI
        likert_score_2 = 0
        if self.ui.radioButton_2_1.isChecked():
            likert_score_2 = 1
        elif self.ui.radioButton_2_2.isChecked():
            likert_score_2 = 2
        elif self.ui.radioButton_2_3.isChecked():
            likert_score_2 = 3
        elif self.ui.radioButton_2_4.isChecked():
            likert_score_2 = 4
        elif self.ui.radioButton_2_5.isChecked():
            likert_score_2 = 5

        self.likert_scores_2 = likert_score_2

        # Save Likert score visibility DCE
        likert_score_3 = 0
        if self.ui.radioButton_3_1.isChecked():
            likert_score_3 = 1
        elif self.ui.radioButton_3_2.isChecked():
            likert_score_3 = 2
        elif self.ui.radioButton_3_3.isChecked():
            likert_score_3 = 3
        elif self.ui.radioButton_3_4.isChecked():
            likert_score_3 = 4
        elif self.ui.radioButton_3_5.isChecked():
            likert_score_3 = 5

        self.likert_scores_3 = likert_score_3

        # Save Residual Tumor
        likert_score_5 = 0
        if self.ui.radioButton_5_1.isChecked():
            likert_score_5 = 1
        elif self.ui.radioButton_5_2.isChecked():
            likert_score_5 = 2

        self.likert_scores_5 = likert_score_5

        # Save Muscle Invasion
        likert_score_6 = 0
        if self.ui.radioButton_6_1.isChecked():
            likert_score_6 = 1
        elif self.ui.radioButton_6_2.isChecked():
            likert_score_6 = 2

        self.likert_scores_6 = likert_score_6

        # Save Radiological downstaging
        likert_score_7 = 0
        if self.ui.radioButton_7_1.isChecked():
            likert_score_7 = 1
        elif self.ui.radioButton_7_2.isChecked():
            likert_score_7 = 2

        self.likert_scores_7 = likert_score_7

        # Save Likert score visibility DCE
        likert_score_8 = 0
        if self.ui.radioButton_8_1.isChecked():
            likert_score_8 = 1
        elif self.ui.radioButton_8_2.isChecked():
            likert_score_8 = 2
        elif self.ui.radioButton_8_3.isChecked():
            likert_score_8 = 3
        elif self.ui.radioButton_8_4.isChecked():
            likert_score_8 = 4
        elif self.ui.radioButton_8_5.isChecked():
            likert_score_8 = 5

        self.likert_scores_8 = likert_score_8

        # Save VIRADS score
        VIRADS = 0
        if self.ui.radioButton_12_1.isChecked():
            VIRADS = 1
        elif self.ui.radioButton_12_2.isChecked():
            VIRADS = 2
        elif self.ui.radioButton_12_3.isChecked():
            VIRADS = 3
        elif self.ui.radioButton_12_4.isChecked():
            VIRADS = 4
        elif self.ui.radioButton_12_5.isChecked():
            VIRADS = 5

        self.VIRADS = VIRADS

        # Save VIRADS score
        nacVIRADS = 0
        if self.ui.radioButton_13_1.isChecked():
            nacVIRADS = 1
        elif self.ui.radioButton_13_2.isChecked():
            nacVIRADS = 2
        elif self.ui.radioButton_13_3.isChecked():
            nacVIRADS = 3
        elif self.ui.radioButton_13_4.isChecked():
            nacVIRADS = 4
        elif self.ui.radioButton_13_5.isChecked():
            nacVIRADS = 5

        self.nacVIRADS = nacVIRADS

    def load_nifti_segmentation(self):
        # Reset the slice views to clear any remaining segmentations
        slicer.util.resetSliceViews()

        file_path = self.nifti_files[self.current_index]
        if self.volume_node:
            slicer.mrmlScene.RemoveNode(self.volume_node)
        if self.segmentation_node:
            slicer.mrmlScene.RemoveNode(self.segmentation_node)
        if self.segmentEditorNode:
            slicer.mrmlScene.RemoveNode(self.segmentEditorNode)

        self.volume_node = slicer.util.loadVolume(file_path)
        slicer.app.applicationLogic().PropagateVolumeSelection(0)


        segmentation_file_path = self.segmentation_files[self.current_index]
        self.segmentation_node = slicer.util.loadSegmentation(segmentation_file_path)
        self.segmentation_node.GetDisplayNode().SetColor(self.segmentation_color)
        self.set_segmentation_and_mask_for_segmentation_editor()

        print(segmentation_file_path)


    def set_segmentation_and_mask_for_segmentation_editor(self):
        slicer.app.processEvents()
        self.segmentEditorWidget.setMRMLScene(slicer.mrmlScene)

        segmentation = self.segmentation_node.GetSegmentation()
        original_name = self.segmentation_node.GetName()

        if segmentation.GetNumberOfSegments() == 1:
            # Set name of segment
            segmentID = segmentation.GetNthSegmentID(0)  # Get the ID of the first and only segment
            segment = segmentation.GetSegment(segmentID)  # Get the segment by its ID
            segment.SetName("Tumor")

        # Check if the current segmentation node is empty
        if segmentation.GetNumberOfSegments() == 0:
            # Create a new segmentation node with an empty segment
            new_segmentation_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
            segment = slicer.vtkSegment()
            segment.SetName("Tumor (empty)")

            new_segmentation_node.GetSegmentation().AddSegment(segment)
            new_segmentation_node.SetName(original_name)
            self.segmentation_node = new_segmentation_node

        segmentID = self.segmentation_node.GetSegmentation().GetNthSegmentID(0)

        self.segmentEditorNode = slicer.vtkMRMLSegmentEditorNode()
        slicer.mrmlScene.AddNode(self.segmentEditorNode)
        self.segmentEditorWidget.setMRMLSegmentEditorNode(self.segmentEditorNode)
        self.segmentEditorWidget.setSegmentationNode(self.segmentation_node)
        self.segmentEditorWidget.setCurrentSegmentID(segmentID)
        self.segmentEditorWidget.setSourceVolumeNode(self.volume_node)

    def resetUIElements(self):
        # Check all dummy radio buttons to effectively uncheck the other buttons in the group
        for dummy_rb in self.dummy_radio_buttons:
            dummy_rb.setChecked(True)

        # Clean text fields
        self.ui.measured_diameter1.setPlainText("")
        self.ui.measured_diameter2.setPlainText("")
        self.ui.measured_diameter3.setPlainText("")
        self.ui.comment_diameter.setPlainText("")
        self.ui.comment.setPlainText("")

        self.coordinates_list_length = []
        self.coordinates_list_width = []
        self.coordinates_list_height = []

        print("All UI elements reset.")

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.GetNodeReference("InputVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        # if inputParameterNode:
        #    self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        self._parameterNode.EndModify(wasModified)


#
# SlicerLikertDLratingLogic
#

class SlicerLikertDLratingLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)


#
# SlicerLikertDLratingTest
#

class SlicerLikertDLratingTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_SlicerLikertDLrating1()

    def test_SlicerLikertDLrating1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        self.delayDisplay('Test passed')
