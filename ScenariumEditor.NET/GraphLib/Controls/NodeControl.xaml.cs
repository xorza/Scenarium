﻿using System;
using System.ComponentModel;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using GraphLib.ViewModel;

namespace GraphLib.Controls;

public partial class NodeControl : UserControl {
    private Node _viewModel = null!;

    public NodeControl() {
        InitializeComponent();
    }

    private void Node_OnLoaded(object sender, RoutedEventArgs e) {
        _viewModel = (Node)DataContext!;
        _viewModel.UpdatePinPositions();
    }

    public event EventHandler<Pin> PinClick = null;
    public event EventHandler DeletePressed = null;
    public event EventHandler Selected = null;


    #region dragging

    private bool _isDragging = false;
    private Point _headerDragMousePosition;

    private void Header_OnMouseLeftButtonDown(object sender, MouseButtonEventArgs e) {
        var header = (FrameworkElement)sender!;
        Selected?.Invoke(this, EventArgs.Empty);
        if (header.CaptureMouse()) {
            _headerDragMousePosition = e.GetPosition(header);
            _isDragging = true;
            e.Handled = true;
        }
    }

    private void Header_OnMouseLeftButtonUp(object sender, MouseButtonEventArgs e) {
        var header = (FrameworkElement)sender!;
        _isDragging = false;
        header.ReleaseMouseCapture();
    }

    private void Header_OnMouseMove(object sender, MouseEventArgs e) {
        if (!_isDragging) return;

        var header = (FrameworkElement)sender!;
        var currentPosition = e.GetPosition(header);
        var delta = currentPosition - _headerDragMousePosition;

        _viewModel.CanvasPosition += delta;
    }

    #endregion

    private void PinButton_OnLoaded(object sender, RoutedEventArgs e) {
        var element = (FrameworkElement)sender!;
        var position = element.TranslatePoint(new Point(element.ActualWidth / 2.0f, element.ActualHeight / 2.0f), this);
        var pin = (Pin)element.DataContext!;
        pin.NodePosition = position;
    }

    private void PinButton_OnClick(object sender, RoutedEventArgs e) {
        var element = (FrameworkElement)sender;
        var pin = (Pin)element.DataContext!;
        PinClick?.Invoke(this, pin);
    }

    private void DeleteButton_OnClick(object sender, RoutedEventArgs e) {
        DeletePressed?.Invoke(this, EventArgs.Empty);
    }

    private void Node_OnMouseDown(object sender, MouseButtonEventArgs e) {
        Selected?.Invoke(this, EventArgs.Empty);
        e.Handled = true;
    }
}