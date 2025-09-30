class AnalysisTask {
  int? id;
  String imagePath;
  int reportId; // ссылается на reports.id (INTEGER)
  String status; // 'pending'|'uploading'|'done'|'failed'
  DateTime createdAt;
  int retries;
  bool onlyOnWifi;

  AnalysisTask({
    this.id,
    required this.imagePath,
    required this.reportId,
    this.status = 'pending',
    DateTime? createdAt,
    this.retries = 0,
    this.onlyOnWifi = false,
  }) : createdAt = createdAt ?? DateTime.now();

  Map<String, dynamic> toMap() => {
        if (id != null) 'id': id,
        'imagePath': imagePath,
        'reportId': reportId,
        'status': status,
        'createdAt': createdAt.toIso8601String(),
        'retries': retries,
        'onlyOnWifi': onlyOnWifi ? 1 : 0,
      };

  factory AnalysisTask.fromMap(Map<String, dynamic> m) => AnalysisTask(
        id: m['id'] as int?,
        imagePath: m['imagePath'] as String,
        reportId: m['reportId'] as int,
        status: m['status'] as String,
        createdAt: DateTime.parse(m['createdAt'] as String),
        retries: (m['retries'] ?? 0) as int,
        onlyOnWifi: ((m['onlyOnWifi'] ?? 0) as int) == 1,
      );
}
