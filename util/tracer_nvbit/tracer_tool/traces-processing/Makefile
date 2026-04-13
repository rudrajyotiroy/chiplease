TARGET := post-traces-processing

$(TARGET): post-traces-processing.cpp
	g++ -std=c++17 -O3 -g -o $@ $^ -lstdc++fs

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o
