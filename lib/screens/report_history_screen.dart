import 'package:flutter/material.dart';
import 'package:plant_analyzer/models/report.dart';
import 'package:plant_analyzer/screens/report_screen.dart';
import 'package:plant_analyzer/services/report_service.dart';

class ReportHistoryScreen extends StatefulWidget {
  @override
  _ReportHistoryScreenState createState() => _ReportHistoryScreenState();
}

class _ReportHistoryScreenState extends State<ReportHistoryScreen> {
  final ReportService _service = ReportService();
  List<Report> _reports = [];
  int _currentPage = 1;
  final int _limit = 10;
  int _totalPages = 1;
  bool _loading = false;
  // Фильтры
  double? _minProbability;
  double? _maxProbability;
  String? _selectedSpecies;
  Map<String, bool> _featureFilters = {
    'trunkRot': false,
    'trunkHoles': false,
    'trunkCracks': false,
    'trunkDamage': false,
    'crownDamage': false,
    'fruitingBodies': false,
    'diseases': false,
  };
  void _openFilters() {
    showModalBottomSheet(
      context: context,
      builder: (context) {
        // StatefulBuilder позволяет локально вызывать setState внутри модалки
        return StatefulBuilder(
          builder: (context, setModalState) {
            return Padding(
              padding: EdgeInsets.all(16),
              child: SingleChildScrollView(
                child: Column(
                  children: [
                    // Min probability
                    TextField(
                      keyboardType: TextInputType.number,
                      decoration: InputDecoration(labelText: 'Min Probability'),
                      onChanged: (val) => setModalState(() {
                        _minProbability = double.tryParse(val);
                      }),
                    ),
                    // Max probability
                    TextField(
                      keyboardType: TextInputType.number,
                      decoration: InputDecoration(labelText: 'Max Probability'),
                      onChanged: (val) => setModalState(() {
                        _maxProbability = double.tryParse(val);
                      }),
                    ),
                    SizedBox(height: 16),
                    // Признаки
                    ..._featureFilters.keys.map((key) {
                      return CheckboxListTile(
                        title: Text(key),
                        value: _featureFilters[key],
                        onChanged: (val) {
                          setModalState(() {
                            _featureFilters[key] = val ?? false;
                          });
                        },
                      );
                    }).toList(),
                    SizedBox(height: 16),
                    ElevatedButton(
                      onPressed: () {
                        Navigator.pop(context);
                        _applyFilters();
                      },
                      child: Text('Применить фильтры'),
                    ),
                  ],
                ),
              ),
            );
          },
        );
      },
    );
  }

  void _applyFilters() {
    _loadReports(page: 1);
  }



  @override
  void initState() {
    super.initState();
    _loadReports();
  }

  Future<void> _loadReports({int page = 1}) async {
  setState(() => _loading = true);

  // Считаем общее количество отчетов по текущим фильтрам
  int totalReports = await _service.countReports(
      minProbability: _minProbability,
      maxProbability: _maxProbability,
      species: _selectedSpecies,
      features: _featureFilters,
    );

    _totalPages = (totalReports / _limit).ceil();
    if (_totalPages == 0) _totalPages = 1; // хотя бы 1 страница

    // Загружаем отчеты для текущей страницы
    List<Report> reports = await _service.fetchReports(
      page: page,
      limit: _limit,
      minProbability: _minProbability,
      maxProbability: _maxProbability,
      species: _selectedSpecies,
      features: _featureFilters,
    );

    setState(() {
      _reports = reports;
      _currentPage = page;
      _loading = false;
    });
  }




  void _nextPage() {
    if (_currentPage < _totalPages) _loadReports(page: _currentPage + 1);
  }

  void _prevPage() {
    if (_currentPage > 1) _loadReports(page: _currentPage - 1);
  }


  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('История отчетов'),
        actions: [
          IconButton(
            icon: Icon(Icons.filter_list),
            onPressed: _openFilters, // открываем фильтры
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: RefreshIndicator(
              onRefresh: () => _loadReports(page: 1),
              child: _loading
                  ? Center(child: CircularProgressIndicator())
                  : ListView.builder(
                      itemCount: _reports.length,
                      itemBuilder: (context, index) {
                        final report = _reports[index];
                        return ListTile(
                          title: Text(report.plantName),
                          subtitle: Text('Вероятность: ${report.probability}%'),
                          onTap: () {
                            Navigator.push(
                              context,
                              MaterialPageRoute(
                                builder: (_) => ReportScreen(report: report),
                              ),
                            );
                          },
                        );
                      },
                    ),
            ),
          ),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              IconButton(onPressed: _prevPage, icon: Icon(Icons.arrow_back)),
              Text('Страница $_currentPage'),
              IconButton(onPressed: _nextPage, icon: Icon(Icons.arrow_forward)),
            ],
          ),
        ],
      ),
    );
  }

}
