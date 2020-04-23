package hw

import (
	"context"
	"testing"

	dataframe "github.com/rocketlaunchr/dataframe-go"
	evalFn "github.com/rocketlaunchr/dataframe-go/forecast/evaluation"
)

func TestHW(t *testing.T) {
	ctx := context.Background()

	// 48 + 24 = 72 data pts + extra 12
	data := dataframe.NewSeriesFloat64("simple data", nil, 30, 21, 29, 31, 40, 48, 53, 47, 37, 39, 31, 29, 17, 9, 20, 24, 27, 35, 41, 38,
		27, 31, 27, 26, 21, 13, 21, 18, 33, 35, 40, 36, 22, 24, 21, 20, 17, 14, 17, 19,
		26, 29, 40, 31, 20, 24, 18, 26, 17, 9, 17, 21, 28, 32, 46, 33, 23, 28, 22, 27,
		18, 8, 17, 21, 31, 34, 44, 38, 31, 30, 26, 32, 45, 34, 30, 27, 25, 22, 28, 33, 42, 32, 40, 52,
	)

	var (
		period int  = 12
		h      uint = 24
	)

	// fmt.Println(data.Table())
	alpha := 0.716
	beta := 0.029
	gamma := 0.993

	cfg := HoltWintersConfig{
		Alpha:    alpha,
		Beta:     beta,
		Gamma:    gamma,
		Period:   period,
		Seasonal: ADD,
	}

	hwModel := NewHoltWinters()

	if err := hwModel.Configure(cfg); err != nil {
		t.Errorf("error encountered: %s\n", err)
	}

	if err := hwModel.Load(ctx, data, &dataframe.Range{End: &[]int{71}[0]}); err != nil {
		t.Errorf("error encountered: %s\n", err)
	}

	hwPredict, err := hwModel.Predict(ctx, h)
	if err != nil {
		t.Errorf("error encountered: %s\n", err)
	}

	expected := dataframe.NewSeriesFloat64("expected", nil,
		22.42511411230803, 15.343371755223066, 24.14282581581347, 27.02259921391996, 35.31139046245393, 38.999014669337356,
		49.243283875692654, 40.84636009563803, 31.205180503707012, 32.96259980122959, 28.5164783238384, 32.30616336737171,
		22.737583867810464, 15.655841510725496, 24.4552955713159, 27.33506896942239, 35.62386021795636, 39.31148442483978,
		49.55575363119508, 41.15882985114047, 31.517650259209443, 33.275069556732014, 28.82894807934083, 32.618633122874144,
	)

	eq, err := hwPredict.IsEqual(ctx, expected)
	if err != nil {
		t.Errorf("error encountered: %s\n", err)
	}
	if !eq {
		t.Errorf("prection: \n%s\n is not equal to expected: \n%s\n", hwPredict.Table(), expected.Table())
	}

	errVal, err := hwModel.Evaluate(ctx, hwPredict, evalFn.SumOfSquaredErrors)
	if err != nil {
		t.Errorf("error encountered: %s", err)
	}
	expSSE := 2437.309198
	_ = expSSE
	_ = errVal
	// if errVal != expSSE {
	// 	t.Errorf("error: expected val: %f not equal to actual err val: %f", expSSE, errVal)
	// }
}
