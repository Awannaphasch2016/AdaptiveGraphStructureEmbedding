# from src.Evaluation.report_performance import ReportPerformance
# from src.Visualization.plot_performance import PlotPerformance
#
# class PerformanceTracker(ReportPerformance, PlotPerformance):
#     def __init__(self,
#                  # PlotPerformance
#                  is_plotted=None,
#                  is_saved_plot=None,
#                  # ReportPerformance
#                  is_displayed_performance_table=None,
#                  is_plotted_roc =None,
#                  is_saved_performance_table=None,
#                  is_saved_plotted_roc=None
#                  ):
#         super(PerformanceTracker, self).__init__(
#             is_plotted=is_plotted,
#             is_saved_plot=is_saved_plot,
#             # ReportPerformance
#             is_displayed_performance_table=is_displayed_performance_table,
#             is_plotted_roc=is_plotted_roc,
#             is_saved_performance_table=is_saved_performance_table,
#             is_saved_plotted_roc=is_saved_plotted_roc
#
#         )
#
#
#         # PlotPerformance
#         self.is_plotted = is_plotted
#         self.is_saved_plot = is_saved_plot
#         # ReportPerformance
#         self.is_displayed_table_performance = is_displayed_performance_table
#         self.is_plotted_roc = is_plotted_roc
#         self.is_saved_table_performance = is_saved_performance_table
#         self.is_saved_plotted_roc = is_saved_plotted_roc
#
#
# if __name__ == '__main__':
#     performance_tracker = PerformanceTracker(True, True,True,True,True,True)
