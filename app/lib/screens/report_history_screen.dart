import 'package:flutter/material.dart';
import '../models/report.dart';
import '../screens/report_screen.dart';
import '../services/report_service.dart';
import '../services/api_service.dart';
import 'dart:io'; // для File
import '../services/connectivity_service.dart';

class ReportHistoryScreen extends StatefulWidget {
  @override
  _ReportHistoryScreenState createState() => _ReportHistoryScreenState();
}

class _ReportHistoryScreenState extends State<ReportHistoryScreen> {
  final ReportService _service = ReportService();
  final ApiService _apiService = ApiService();
  List<Report> _reports = [];
  int _currentPage = 1;
  final int _limit = 20;
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
  // словарь переводов
  final Map<String, String> _featureLabels = {
    'trunkRot': 'Гниль ствола',
    'trunkHoles': 'Дупла',
    'trunkCracks': 'Трещины',
    'trunkDamage': 'Повреждение ствола',
    'crownDamage': 'Повреждение кроны',
    'fruitingBodies': 'Плодовые тела грибов',
    'diseases': 'Болезни',
  };

  void _openFilters() async {
    final hasInternet = await ConnectivityService.hasInternet();

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
                    // Поля мин/макс вероятности отображаем только если нет интернета
                    if (!hasInternet) ...[
                      TextField(
                        keyboardType: TextInputType.number,
                        decoration: InputDecoration(
                          labelText: 'Мин вероятность детекции растения',
                        ),
                        onChanged: (val) => setModalState(() {
                          _minProbability = double.tryParse(val);
                        }),
                      ),
                      SizedBox(height: 8),
                      TextField(
                        keyboardType: TextInputType.number,
                        decoration: InputDecoration(
                          labelText: 'Макс вероятность детекции растения',
                        ),
                        onChanged: (val) => setModalState(() {
                          _maxProbability = double.tryParse(val);
                        }),
                      ),
                      SizedBox(height: 16),
                    ],

                    // Признаки
                    ..._featureFilters.keys.map((key) {
                      return CheckboxListTile(
                        title: Text(_featureLabels[key] ?? key),
                        value: _featureFilters[key],
                        onChanged: (val) {
                          setModalState(() {
                            _featureFilters[key] = val ?? false;
                          });
                        },
                      );
                    }).toList(),

                    SizedBox(height: 16),
                    // Кнопки
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      children: [
                        ElevatedButton(
                          onPressed: () {
                            Navigator.pop(context);
                            _applyFilters();
                          },
                          child: Text('Применить фильтры'),
                        ),
                        ElevatedButton(
                          onPressed: () {
                            setModalState(() {
                              _minProbability = null;
                              _maxProbability = null;
                              _selectedSpecies = null;
                              _featureFilters.updateAll((key, value) => false);
                            });
                            Navigator.pop(context);
                            _applyFilters();
                          },
                          child: Text('Сбросить'),
                        ),
                      ],
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
  void _resetFilters() {
  setState(() {
    _minProbability = null;
    _maxProbability = null;
    _selectedSpecies = null;
    _featureFilters.updateAll((key, value) => false);
  });
}



  @override
  void initState() {
    super.initState();
    _loadReports(); // универсальный метод
  }


  /// Метод загрузки отчетов без интернета
  Future<void> _loadOfflineReports({int page = 1}) async {
    setState(() => _loading = true);

    int totalReports = await _service.countReports(
      minProbability: _minProbability,
      maxProbability: _maxProbability,
      species: _selectedSpecies,
      features: _featureFilters,
    );

    _totalPages = (totalReports / _limit).ceil();
    if (_totalPages == 0) _totalPages = 1;

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
  Future<void> _loadOnlineReports({int page = 1}) async {
    setState(() => _loading = true);

    List<Report> reports = await _apiService.fetchReports(
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

  void _nextPage() async {
    await _loadReports(page: _currentPage + 1);
    if (_reports.isEmpty) {
      // вернулись пустые данные → остаёмся на старой странице
      _currentPage--;
      await _loadReports(page: _currentPage);
    }
  }

  void _prevPage() {
    if (_currentPage > 1) _loadReports(page: _currentPage - 1);
  }



  /// общий метод _loadReports как обертка для реализации с интернетом и без
  Future<void> _loadReports({int page = 1}) async {
    final internetAvailable = await ConnectivityService.hasInternet();
    // здесь позже можно проверить интернет и вызвать онлайн/офлайн версии
    if (internetAvailable){
      await _loadOnlineReports(page: page);

    }
    else{
      await _loadOfflineReports(page: page);
    }
    
  }

  /*void _nextPage() {
    if (_currentPage < _totalPages) _loadReports(page: _currentPage + 1);
  }

  void _prevPage() {
    if (_currentPage > 1) _loadReports(page: _currentPage - 1);
  }*/


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
                        leading: FutureBuilder<bool>(
                          future: ConnectivityService.hasInternet(), // твой сервис
                          builder: (context, snapshot) {
                            final hasInternet = snapshot.data ?? false;

                            if (hasInternet &&
                                report.imageUrl != null &&
                                report.imageUrl!.isNotEmpty) {
                              return Image.network(
                                report.imageUrl!,
                                width: 50,
                                height: 50,
                                fit: BoxFit.cover,
                              );
                            } else if (report.imagePath != null &&
                              report.imagePath!.isNotEmpty &&
                              report.imagePath != 'no') {
                            return Image.file(
                              File(report.imagePath!),
                              width: 50,
                              height: 50,
                              fit: BoxFit.cover,
                            );
                            } else {
                              return const Icon(
                                Icons.image_not_supported,
                                size: 50,
                              );
                            }
                          },
                        ),


                        title: Text(report.plantName ?? 'Неизвестное растение'),
                        subtitle: Text('Вероятность: ${report.probability ?? 0}%'),
                        onTap: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (_) => ReportScreen(reports: [report]),
                            ),
                          );
                        },

                        trailing: FutureBuilder<bool>(
                          future: ConnectivityService.hasInternet(),
                          builder: (context, snapshot) {
                            final hasInternet = snapshot.data ?? false;

                            if (hasInternet) {
                              return const SizedBox.shrink(); // пустое место вместо null
                            } else {
                              return IconButton(
                                icon: const Icon(Icons.delete, color: Color.fromARGB(255, 44, 85, 44)),
                                onPressed: () async {
                                  if (report.id != null) {
                                    await _service.deleteReportById(report.id!);
                                    _loadReports(page: _currentPage); // обновляем список
                                  }
                                },
                              );
                            }
                          },
                        ),

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
